"""PPO with a Beta policy on [0,1], written from scratch; optional torch.distributed (gloo) data parallelism.

Each worker collects its own episodes; gradients are averaged with all_reduce after every minibatch backward,
and advantage normalisation uses global (all-worker) moments. The same code runs single-process (no
torch.distributed initialised), multi-process on one machine, or multi-node under torchrun.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from scipy.special import betaincinv
from torch.distributions import Beta

ROLLOUT_PARTITIONS = 8

HP = dict(hidden=64, episodes_per_update=64, L=52, updates=250, gamma=0.95, lam=0.95, clip=0.2, epochs=8,
          minibatches=4, lr=3e-4, vf_coef=0.5, ent_coef=0.001, max_grad_norm=0.5, risk_aversion=2.0)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class ActorCritic(nn.Module):
    def __init__(self, n_obs, hidden=64):
        super().__init__()
        self.pi = mlp(n_obs, hidden, 2); self.v = mlp(n_obs, hidden, 1)

    def dist(self, x):
        ab = nn.functional.softplus(self.pi(x)) + 1.0
        return Beta(ab[..., 0], ab[..., 1])

    def value(self, x):
        return self.v(x).squeeze(-1)

    @torch.no_grad()
    def mean_action(self, obs: np.ndarray) -> float:
        d = self.dist(torch.as_tensor(obs, dtype=next(self.parameters()).dtype))
        return float(d.mean)


def _ddp():
    return dist.is_available() and dist.is_initialized()


def partition_mean(rows):
    """Average logical partitions in a fixed order, independent of the process count."""
    if _ddp():
        dist.all_reduce(rows, op=dist.ReduceOp.SUM)
    return rows.mean(dim=0)


def sync_params(model):
    if _ddp():
        for p in model.parameters():
            dist.broadcast(p.detach(), src=0)


def train(env_factory, n_obs, seed, hp=HP, log=None):
    """env_factory(B, rng) -> BatchEnv. Returns (model, history list of dicts)."""
    rank = dist.get_rank() if _ddp() else 0
    world = dist.get_world_size() if _ddp() else 1
    if ROLLOUT_PARTITIONS % world or hp["episodes_per_update"] % ROLLOUT_PARTITIONS:
        raise ValueError("The process count must divide the eight logical rollout partitions")
    partitions = list(range(rank, ROLLOUT_PARTITIONS, world))
    torch.manual_seed(seed)
    model = ActorCritic(n_obs, hp["hidden"]).double()
    sync_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], eps=1e-5)
    L = hp["L"]
    B = hp["episodes_per_update"] // ROLLOUT_PARTITIONS
    start_rng = [np.random.default_rng([seed, part, 0]) for part in partitions]
    action_rng = [np.random.default_rng([seed, part, 1]) for part in partitions]
    envs = [env_factory(B, rng) for rng in start_rng]
    total_samples = L * B
    mb = total_samples // hp["minibatches"]
    if total_samples % hp["minibatches"]:
        raise ValueError("Each partition must split into equal-sized minibatches")
    params = list(model.parameters())
    sizes = [p.numel() for p in params]
    gradients = torch.zeros((ROLLOUT_PARTITIONS, sum(sizes)), dtype=torch.float64)
    moments = torch.zeros((ROLLOUT_PARTITIONS, 2), dtype=torch.float64)
    update_stats = torch.zeros((ROLLOUT_PARTITIONS, 5), dtype=torch.float64)
    hist = []
    for u in range(hp["updates"]):
        trajectories = []
        moments.zero_()
        update_stats.zero_()
        with torch.no_grad():
            for part, env, rng in zip(partitions, envs, action_rng):
                obs = np.empty((L + 1, B, n_obs))
                act = np.empty((L, B))
                logp = np.empty((L, B))
                rew = np.empty((L, B))
                uniforms = rng.random((L, B))
                obs[0] = env.reset()
                for k in range(L):
                    d = model.dist(torch.from_numpy(obs[k]))
                    sampled = betaincinv(d.concentration1.numpy(), d.concentration0.numpy(), uniforms[k])
                    a = torch.from_numpy(sampled).clamp(1e-6, 1 - 1e-6)
                    act[k] = a.numpy()
                    logp[k] = d.log_prob(a).numpy()
                    obs[k + 1], rew[k], _ = env.step(act[k])
                val = model.value(torch.from_numpy(obs)).numpy()
                adv = np.empty((L, B))
                g = np.zeros(B)
                for k in reversed(range(L)):
                    delta = rew[k] + hp["gamma"] * val[k + 1] - val[k]
                    g = delta + hp["gamma"] * hp["lam"] * g
                    adv[k] = g
                ret = adv + val[:L]
                moments[part] = torch.tensor([adv.mean(), (adv ** 2).mean()], dtype=torch.float64)
                update_stats[part, :2] = torch.tensor([rew.sum(0).mean(), act.mean()], dtype=torch.float64)
                trajectories.append((torch.from_numpy(obs[:L].reshape(-1, n_obs)),
                                     torch.from_numpy(act.flatten()), torch.from_numpy(logp.flatten()),
                                     torch.from_numpy(ret.flatten()), adv.flatten()))

        mom = partition_mean(moments)
        a_mu = float(mom[0])
        a_sd = float(np.sqrt(max(mom[1] - mom[0] ** 2, 1e-12)))
        batches = [(O, A, LP, RET, torch.from_numpy((adv - a_mu) / a_sd))
                   for O, A, LP, RET, adv in trajectories]
        permutation_rng = [np.random.default_rng([seed, part, 2, u]) for part in partitions]
        for _ in range(hp["epochs"]):
            orders = [rng.permutation(total_samples).reshape(hp["minibatches"], mb) for rng in permutation_rng]
            for i in range(hp["minibatches"]):
                gradients.zero_()
                for part, order, (O, A, LP, RET, ADV) in zip(partitions, orders, batches):
                    idx = order[i]
                    d = model.dist(O[idx])
                    lp = d.log_prob(A[idx])
                    ratio = torch.exp(lp - LP[idx])
                    pg = -torch.min(ratio * ADV[idx], ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * ADV[idx]).mean()
                    vl = 0.5 * ((model.value(O[idx]) - RET[idx]) ** 2).mean()
                    ent = d.entropy().mean()
                    loss = pg + hp["vf_coef"] * vl - hp["ent_coef"] * ent
                    grads = torch.autograd.grad(loss, params)
                    gradients[part] = torch.cat([grad.flatten() for grad in grads])
                    update_stats[part, 2:] += torch.tensor([pg.item(), vl.item(), ent.item()])
                mean_grad = partition_mean(gradients)
                for p, grad in zip(params, mean_grad.split(sizes)):
                    p.grad = grad.reshape_as(p)
                nn.utils.clip_grad_norm_(params, hp["max_grad_norm"])
                opt.step()
        m = partition_mean(update_stats).tolist()
        pg_loss, v_loss, entropy = [x / (hp["epochs"] * hp["minibatches"]) for x in m[2:]]
        hist.append({"update": u, "ep_reward": m[0], "mean_eta": m[1], "pg_loss": pg_loss,
                     "v_loss": v_loss, "entropy": entropy})
        if log and rank == 0 and (u % 50 == 0 or u == hp["updates"] - 1):
            log(f"  update {u:4d} ep_reward {m[0]:+.4f} mean_eta {m[1]:.3f} v_loss {v_loss:.4f}")
    return model, hist
