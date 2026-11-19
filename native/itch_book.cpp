// NASDAQ TotalView-ITCH 5.0 parser and displayed order book for a list of symbols (design/tick_costs_plan.md).
//
// Reads the decompressed binary stream (2-byte big-endian length, then the message) from standard input:
//   gzip -dc S120825-v50.txt.gz | itch_book --symbols SPY,QQQ --out tmp/tick/2025-12-08
// Writes, under the --out prefix:
//   _summary.csv  one row per symbol: time-weighted quoted half-spread over 9:30-16:00 and 9:30-9:35, depth at the
//                 best quotes, executions against displayed and non-displayed orders, opening and closing crosses
//   _trades.csv   every execution of a listed symbol: time, kind, price, shares, sign (+1 incoming buy, -1 incoming
//                 sell, 0 unknown), and NASDAQ's best bid and ask just before it
//   _mids.csv     NASDAQ midpoint of each symbol at every whole second from 9:30:00 to 16:00:00 (empty if none)
//   _checks.csv   message counts by type, length mismatches, unknown order references, negative levels, crossed books
// Exit status 3 when a check of the plan fails (a length mismatch, an unknown reference or a negative level).
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr std::int64_t NS = 1000000000LL;
constexpr std::int64_t OPEN = 34200 * NS;     // 9:30:00 Eastern
constexpr std::int64_t OPEN5 = OPEN + 300 * NS;
constexpr std::int64_t CLOSE = 57600 * NS;    // 16:00:00
constexpr int GRID = 23401;                   // whole seconds from 9:30:00 to 16:00:00 inclusive

std::uint64_t be(const unsigned char* p, int n) {
  std::uint64_t v = 0;
  for (int i = 0; i < n; ++i) v = (v << 8) | p[i];
  return v;
}

int expected_length(char type) {
  switch (type) {
    case 'S': return 12;
    case 'R': return 39;
    case 'A': return 36;
    case 'F': return 40;
    case 'E': return 31;
    case 'C': return 36;
    case 'X': return 23;
    case 'D': return 19;
    case 'U': return 35;
    case 'P': return 44;
    case 'Q': return 40;
    case 'B': return 19;
    default: return -1;                       // not read by this parser
  }
}

struct Order {
  int sym;
  char side;
  std::uint32_t price;
  std::uint32_t shares;
};

struct Book {
  std::string name;
  std::map<std::uint32_t, std::int64_t> bids, asks;
  std::int64_t last = 0;                      // time of the last book change
  double valid_s = 0, spread_s = 0, open5_valid_s = 0, open5_spread_s = 0, bid_depth_s = 0, ask_depth_s = 0;
  double crossed_s = 0;
  std::int64_t displayed_execs = 0, displayed_shares = 0, hidden_execs = 0, hidden_shares = 0;
  double open_cross_price = NAN, close_cross_price = NAN;
  std::int64_t open_cross_shares = 0, close_cross_shares = 0;
  std::vector<double> mids = std::vector<double>(GRID, NAN);
  int grid_next = 0;

  bool two_sided() const { return !bids.empty() && !asks.empty(); }
  std::uint32_t bid() const { return bids.empty() ? 0 : bids.rbegin()->first; }
  std::uint32_t ask() const { return asks.empty() ? 0 : asks.begin()->first; }

  // Add the current state's contribution over [last, t) and fill grid seconds in that interval.
  void advance(std::int64_t t) {
    if (t <= last) return;
    const bool ok = two_sided() && ask() > bid();
    const double mid = ok ? (static_cast<double>(bid()) + ask()) / 2e4 : NAN;
    auto overlap = [&](std::int64_t a, std::int64_t b) {
      const std::int64_t lo = last > a ? last : a, hi = t < b ? t : b;
      return hi > lo ? static_cast<double>(hi - lo) / NS : 0.0;
    };
    const double in_session = overlap(OPEN, CLOSE);
    if (in_session > 0) {
      if (ok) {
        const double half_bps = (static_cast<double>(ask()) - bid()) / 2e4 / mid * 1e4;
        valid_s += in_session;
        spread_s += half_bps * in_session;
        bid_depth_s += bids.rbegin()->second * (bid() / 1e4) * in_session;
        ask_depth_s += asks.begin()->second * (ask() / 1e4) * in_session;
        const double first5 = overlap(OPEN, OPEN5);
        open5_valid_s += first5;
        open5_spread_s += half_bps * first5;
      } else if (two_sided()) {
        crossed_s += in_session;
      }
    }
    while (grid_next < GRID && OPEN + grid_next * NS < t) {
      if (OPEN + grid_next * NS >= last) mids[grid_next] = mid;
      ++grid_next;
    }
    last = t;
  }
};

struct Counts {
  std::array<std::int64_t, 256> by_type{};
  std::array<std::int64_t, 256> bad_length{};
  std::int64_t unknown_ref = 0, negative_level = 0, crossed_after_message = 0;
};

void change(std::map<std::uint32_t, std::int64_t>& side, std::uint32_t price, std::int64_t delta, Counts& c) {
  auto it = side.find(price);
  const std::int64_t now = (it == side.end() ? 0 : it->second) + delta;
  if (now < 0) ++c.negative_level;
  if (now <= 0) {
    if (it != side.end()) side.erase(it);
  } else if (it == side.end()) {
    side.emplace(price, now);
  } else {
    it->second = now;
  }
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char ch : s) {
    if (ch == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); } else { cur += ch; }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string symbols_arg, out;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--symbols")) symbols_arg = argv[i + 1];
    else if (!std::strcmp(argv[i], "--out")) out = argv[i + 1];
  }
  if (symbols_arg.empty() || out.empty()) {
    std::fprintf(stderr, "usage: itch_book --symbols A,B,C --out prefix < decompressed.itch\n");
    return 2;
  }
  std::vector<Book> books;
  std::unordered_map<std::string, int> wanted;
  for (const auto& name : split(symbols_arg)) {
    wanted.emplace(name, static_cast<int>(books.size()));
    books.push_back(Book{});
    books.back().name = name;
  }
  std::vector<int> locate_to_sym(65536, -1);
  std::unordered_map<std::uint64_t, Order> orders;
  orders.reserve(1 << 22);
  Counts c;

  std::FILE* trades = std::fopen((out + "_trades.csv").c_str(), "w");
  if (!trades) { std::perror("trades file"); return 2; }
  std::fprintf(trades, "symbol,ts_ns,kind,price,shares,sign,bid,ask\n");
  static char buffer[1 << 22];
  std::setvbuf(stdin, buffer, _IOFBF, sizeof(buffer));

  unsigned char len_bytes[2], msg[65536];
  auto record = [&](int sym, std::int64_t ts, char kind, std::uint32_t price, std::int64_t shares, int sign) {
    const Book& b = books[sym];
    std::fprintf(trades, "%s,%lld,%c,%.4f,%lld,%d,%.4f,%.4f\n", b.name.c_str(), static_cast<long long>(ts), kind,
                 price / 1e4, static_cast<long long>(shares), sign, b.bid() / 1e4, b.ask() / 1e4);
  };

  while (std::fread(len_bytes, 1, 2, stdin) == 2) {
    const int n = static_cast<int>(be(len_bytes, 2));
    if (n == 0 || std::fread(msg, 1, n, stdin) != static_cast<std::size_t>(n)) break;
    const char type = static_cast<char>(msg[0]);
    ++c.by_type[static_cast<unsigned char>(type)];
    const int want = expected_length(type);
    if (want < 0) continue;
    if (n != want) { ++c.bad_length[static_cast<unsigned char>(type)]; continue; }
    const int locate = static_cast<int>(be(msg + 1, 2));
    const std::int64_t ts = static_cast<std::int64_t>(be(msg + 5, 6));

    if (type == 'R') {
      std::string name(reinterpret_cast<const char*>(msg + 11), 8);
      name.erase(name.find_last_not_of(' ') + 1);
      auto it = wanted.find(name);
      if (it != wanted.end()) locate_to_sym[locate] = it->second;
      continue;
    }
    const int sym = locate_to_sym[locate];
    if (sym < 0) continue;
    Book& book = books[sym];

    switch (type) {
      case 'A':
      case 'F': {
        const std::uint64_t ref = be(msg + 11, 8);
        const char side = static_cast<char>(msg[19]);
        const auto shares = static_cast<std::uint32_t>(be(msg + 20, 4));
        const auto price = static_cast<std::uint32_t>(be(msg + 32, 4));
        book.advance(ts);
        orders[ref] = Order{sym, side, price, shares};
        change(side == 'B' ? book.bids : book.asks, price, shares, c);
        break;
      }
      case 'E':
      case 'C': {
        const std::uint64_t ref = be(msg + 11, 8);
        const auto shares = static_cast<std::uint32_t>(be(msg + 19, 4));
        auto it = orders.find(ref);
        if (it == orders.end()) { ++c.unknown_ref; break; }
        Order& o = it->second;
        book.advance(ts);
        const bool printable = type == 'E' || msg[31] == 'Y';
        const std::uint32_t price = type == 'E' ? o.price : static_cast<std::uint32_t>(be(msg + 32, 4));
        if (printable) {
          record(sym, ts, type, price, shares, o.side == 'B' ? -1 : 1);
          if (ts >= OPEN && ts < CLOSE) { ++book.displayed_execs; book.displayed_shares += shares; }
        }
        change(o.side == 'B' ? book.bids : book.asks, o.price, -static_cast<std::int64_t>(shares), c);
        if (shares >= o.shares) orders.erase(it); else o.shares -= shares;
        break;
      }
      case 'X': {
        const std::uint64_t ref = be(msg + 11, 8);
        const auto shares = static_cast<std::uint32_t>(be(msg + 19, 4));
        auto it = orders.find(ref);
        if (it == orders.end()) { ++c.unknown_ref; break; }
        Order& o = it->second;
        book.advance(ts);
        change(o.side == 'B' ? book.bids : book.asks, o.price, -static_cast<std::int64_t>(shares), c);
        if (shares >= o.shares) orders.erase(it); else o.shares -= shares;
        break;
      }
      case 'D': {
        const std::uint64_t ref = be(msg + 11, 8);
        auto it = orders.find(ref);
        if (it == orders.end()) { ++c.unknown_ref; break; }
        book.advance(ts);
        change(it->second.side == 'B' ? book.bids : book.asks, it->second.price,
               -static_cast<std::int64_t>(it->second.shares), c);
        orders.erase(it);
        break;
      }
      case 'U': {
        const std::uint64_t old_ref = be(msg + 11, 8), new_ref = be(msg + 19, 8);
        const auto shares = static_cast<std::uint32_t>(be(msg + 27, 4));
        const auto price = static_cast<std::uint32_t>(be(msg + 31, 4));
        auto it = orders.find(old_ref);
        if (it == orders.end()) { ++c.unknown_ref; break; }
        const Order old = it->second;
        book.advance(ts);
        auto& side = old.side == 'B' ? book.bids : book.asks;
        change(side, old.price, -static_cast<std::int64_t>(old.shares), c);
        orders.erase(it);
        orders[new_ref] = Order{sym, old.side, price, shares};
        change(side, price, shares, c);
        break;
      }
      case 'P': {
        const auto shares = static_cast<std::uint32_t>(be(msg + 20, 4));
        const auto price = static_cast<std::uint32_t>(be(msg + 32, 4));
        book.advance(ts);
        record(sym, ts, 'P', price, shares, 0);
        if (ts >= OPEN && ts < CLOSE) { ++book.hidden_execs; book.hidden_shares += shares; }
        break;
      }
      case 'Q': {
        const auto shares = static_cast<std::int64_t>(be(msg + 11, 8));
        const double price = be(msg + 27, 4) / 1e4;
        const char cross = static_cast<char>(msg[39]);
        if (cross == 'O') { book.open_cross_price = price; book.open_cross_shares = shares; }
        if (cross == 'C') { book.close_cross_price = price; book.close_cross_shares = shares; }
        break;
      }
      default:
        break;
    }
    if (book.two_sided() && book.bid() >= book.ask()) ++c.crossed_after_message;
  }
  std::fclose(trades);

  std::FILE* summary = std::fopen((out + "_summary.csv").c_str(), "w");
  std::FILE* mids = std::fopen((out + "_mids.csv").c_str(), "w");
  if (!summary || !mids) { std::perror("output"); return 2; }
  std::fprintf(summary, "symbol,valid_seconds,quoted_half_spread_bps,open5_valid_seconds,open5_quoted_half_spread_bps,"
                        "bid_depth_usd,ask_depth_usd,crossed_or_locked_seconds,displayed_execs,displayed_shares,"
                        "hidden_execs,hidden_shares,open_cross_price,open_cross_shares,close_cross_price,"
                        "close_cross_shares\n");
  std::fprintf(mids, "symbol,second,mid\n");
  for (Book& b : books) {
    b.advance(CLOSE + NS);
    auto per = [](double total, double seconds) { return seconds > 0 ? total / seconds : NAN; };
    std::fprintf(summary, "%s,%.3f,%.9f,%.3f,%.9f,%.4f,%.4f,%.3f,%lld,%lld,%lld,%lld,%.4f,%lld,%.4f,%lld\n",
                 b.name.c_str(), b.valid_s, per(b.spread_s, b.valid_s), b.open5_valid_s,
                 per(b.open5_spread_s, b.open5_valid_s), per(b.bid_depth_s, b.valid_s), per(b.ask_depth_s, b.valid_s),
                 b.crossed_s, static_cast<long long>(b.displayed_execs), static_cast<long long>(b.displayed_shares),
                 static_cast<long long>(b.hidden_execs), static_cast<long long>(b.hidden_shares), b.open_cross_price,
                 static_cast<long long>(b.open_cross_shares), b.close_cross_price,
                 static_cast<long long>(b.close_cross_shares));
    for (int g = 0; g < GRID; ++g) {
      if (std::isfinite(b.mids[g])) std::fprintf(mids, "%s,%d,%.5f\n", b.name.c_str(), g, b.mids[g]);
    }
  }
  std::fclose(summary);
  std::fclose(mids);

  std::FILE* checks = std::fopen((out + "_checks.csv").c_str(), "w");
  if (!checks) { std::perror("checks"); return 2; }
  std::fprintf(checks, "item,value\n");
  std::int64_t bad = 0;
  for (int t = 0; t < 256; ++t) {
    if (c.by_type[t]) std::fprintf(checks, "messages_%c,%lld\n", t, static_cast<long long>(c.by_type[t]));
    if (c.bad_length[t]) std::fprintf(checks, "bad_length_%c,%lld\n", t, static_cast<long long>(c.bad_length[t]));
    bad += c.bad_length[t];
  }
  std::fprintf(checks, "bad_length_total,%lld\nunknown_reference,%lld\nnegative_level,%lld\ncrossed_after_message,%lld\n",
               static_cast<long long>(bad), static_cast<long long>(c.unknown_ref),
               static_cast<long long>(c.negative_level), static_cast<long long>(c.crossed_after_message));
  std::fclose(checks);
  return (bad || c.unknown_ref || c.negative_level) ? 3 : 0;
}
