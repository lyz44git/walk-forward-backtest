// C++17 kernels for two path-dependent loops of the backtest.
//
//   qmb_sb_indices / qmb_sb_sharpe   stationary bootstrap (Politis and Romano 1994): the resampled
//                                    index path of each draw, and its annualized Sharpe ratio
//   qmb_simulate                     weekly portfolio accounting with drift, partial adjustment or a
//                                    no-trade band, trading costs and borrow fees
//
// Built as a shared library with a C ABI and called from NumPy through ctypes (src/qmb/native.py).
// The Python implementations in qmb.stats and qmb.rebalance.env remain the reference. The random
// numbers of the bootstrap are drawn by NumPy in the reference order, so both implementations
// resample the same weeks; tests require agreement to 1e-12.
//
// Compile without -ffast-math and with -ffp-contract=off: every operation is then an IEEE 754
// double operation in the order written here, on every compiler and CPU.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int OK = 0;
constexpr int BAD_ARGUMENT = 1;
constexpr int INDEX_OUT_OF_RANGE = 2;

// Neumaier compensated sum: the running error of each addition is kept and added back at the end.
class CompensatedSum {
 public:
  void add(double v) {
    const double t = sum_ + v;
    correction_ += (std::fabs(sum_) >= std::fabs(v)) ? (sum_ - t) + v : (v - t) + sum_;
    sum_ = t;
  }
  double value() const { return sum_ + correction_; }

 private:
  double sum_ = 0.0;
  double correction_ = 0.0;
};

bool indices_valid(const std::int64_t* a, std::int64_t count, std::int64_t n) {
  for (std::int64_t i = 0; i < count; ++i) {
    if (a[i] < 0 || a[i] >= n) return false;
  }
  return true;
}

// Next position of a stationary-bootstrap path: restart at a random week with probability p,
// otherwise continue to the following week, wrapping at the end of the sample.
inline std::int64_t next_index(std::int64_t current, bool restart, std::int64_t jump, std::int64_t n) {
  return restart ? jump : (current + 1 == n ? 0 : current + 1);
}

double sharpe(const std::vector<double>& x, double per_year) {
  const auto n = static_cast<double>(x.size());
  CompensatedSum total;
  for (double v : x) total.add(v);
  const double mean = total.value() / n;
  CompensatedSum squares;
  for (double v : x) squares.add((v - mean) * (v - mean));
  const double sd = std::sqrt(squares.value() / (n - 1.0));
  if (!(sd > 0.0)) return std::numeric_limits<double>::quiet_NaN();
  return mean / sd * std::sqrt(per_year);
}

}  // namespace

extern "C" {

// Index paths of `draws` stationary-bootstrap resamples of a series of length n.
// start: (draws); restart, jump: (draws, n) row-major, column 0 unused; out: (draws, n).
int qmb_sb_indices(std::int64_t n, std::int64_t draws, const std::int64_t* start,
                   const std::uint8_t* restart, const std::int64_t* jump, std::int64_t* out) {
  if (n < 1 || draws < 0 || !start || !restart || !jump || !out) return BAD_ARGUMENT;
  if (!indices_valid(start, draws, n) || !indices_valid(jump, draws * n, n)) return INDEX_OUT_OF_RANGE;
  for (std::int64_t d = 0; d < draws; ++d) {
    const std::uint8_t* new_block = restart + d * n;
    const std::int64_t* target = jump + d * n;
    std::int64_t* path = out + d * n;
    path[0] = start[d];
    for (std::int64_t i = 1; i < n; ++i) path[i] = next_index(path[i - 1], new_block[i] != 0, target[i], n);
  }
  return OK;
}

// Annualized Sharpe ratio of each resample of x (length n). Arguments as in qmb_sb_indices; out: (draws).
int qmb_sb_sharpe(const double* x, std::int64_t n, std::int64_t draws, const std::int64_t* start,
                  const std::uint8_t* restart, const std::int64_t* jump, double per_year, double* out) {
  if (n < 2 || draws < 0 || !x || !start || !restart || !jump || !out) return BAD_ARGUMENT;
  if (!indices_valid(start, draws, n) || !indices_valid(jump, draws * n, n)) return INDEX_OUT_OF_RANGE;
  std::vector<double> sample(static_cast<std::size_t>(n));
  for (std::int64_t d = 0; d < draws; ++d) {
    const std::uint8_t* new_block = restart + d * n;
    const std::int64_t* target = jump + d * n;
    std::int64_t current = start[d];
    sample[0] = x[current];
    for (std::int64_t i = 1; i < n; ++i) {
      current = next_index(current, new_block[i] != 0, target[i], n);
      sample[static_cast<std::size_t>(i)] = x[current];
    }
    out[d] = sharpe(sample, per_year);
  }
  return OK;
}

// Weekly accounting of a long-short portfolio over weeks t0 .. t1-1, starting from the full target of
// week t0-1 (qmb.rebalance.env.simulate without a policy).
//
//   d_t = h_{t-1} (1 + r_{t-1}) / (1 + h_{t-1} . r_{t-1})                 weights after one week of drift
//   mode 0, partial adjustment:  h_t = d_t + eta_t (W_t - d_t)
//   mode 1, no-trade band:       h_t = W_t + clip(d_t - W_t, -b_t, b_t)
//   assets absent in week t (M = 0) are closed
//   turnover_t = |h_t - d_t|_1,  gross_t = h_t . r_t,  short_t = -sum of negative h_t
//   net_t = gross_t - turnover_t * cost_bps / 1e4 - short_t * borrow_bps / 1e4 / 52
//
// W, R: (T, N) row-major doubles; M: (T, N) bytes; param: (t1 - t0) values of eta_t or b_t;
// out: (t1 - t0, 4) with columns net, gross, turnover, short; h_final: (N) or null.
int qmb_simulate(std::int64_t T, std::int64_t N, const double* W, const double* R, const std::uint8_t* M,
                 std::int64_t t0, std::int64_t t1, int mode, const double* param, double cost_bps,
                 double borrow_bps, double* out, double* h_final) {
  if (T < 1 || N < 1 || !W || !R || !M || !param || !out) return BAD_ARGUMENT;
  if (t0 < 1 || t1 < t0 || t1 > T || (mode != 0 && mode != 1)) return BAD_ARGUMENT;
  const auto width = static_cast<std::size_t>(N);
  std::vector<double> h(W + (t0 - 1) * N, W + t0 * N);       // holdings set at the previous entry
  std::vector<double> d(width);
  const double* r_prev = R + (t0 - 1) * N;
  for (std::int64_t t = t0; t < t1; ++t) {
    const double* w = W + t * N;
    const double* r = R + t * N;
    const std::uint8_t* present = M + t * N;
    const double p = param[t - t0];

    CompensatedSum growth;
    for (std::size_t i = 0; i < width; ++i) growth.add(h[i] * r_prev[i]);
    const double g = 1.0 + growth.value();
    for (std::size_t i = 0; i < width; ++i) d[i] = h[i] * (1.0 + r_prev[i]) / g;

    CompensatedSum turnover, gross, short_leg;
    for (std::size_t i = 0; i < width; ++i) {
      double target = 0.0;
      if (present[i]) {
        if (mode == 0) {
          target = d[i] + p * (w[i] - d[i]);
        } else {
          double gap = d[i] - w[i];
          if (gap > p) gap = p;
          if (gap < -p) gap = -p;
          target = w[i] + gap;
        }
      }
      h[i] = target;
      turnover.add(std::fabs(target - d[i]));
      gross.add(target * r[i]);
      if (target < 0.0) short_leg.add(-target);
    }
    double* row = out + (t - t0) * 4;
    row[1] = gross.value();
    row[2] = turnover.value();
    row[3] = short_leg.value();
    row[0] = row[1] - row[2] * cost_bps / 1e4 - row[3] * borrow_bps / 1e4 / 52.0;
    r_prev = r;
  }
  if (h_final) {
    for (std::size_t i = 0; i < width; ++i) h_final[i] = h[i];
  }
  return OK;
}

}  // extern "C"
