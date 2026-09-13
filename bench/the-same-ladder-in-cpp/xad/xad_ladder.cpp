/*
 * XAD (C++, AGPL-3.0-or-later, self-serve `git clone` + CMake FetchContent,
 * no license gate) driven directly through its own idiomatic adjoint-tape
 * pattern -- the same per-path clearAll()/registerInput()/newRecording()
 * loop as auto-differentiation/xad's own official benchmark
 * (ad-benchmarks/xad/heston_xad.cpp). No threading in XAD itself, and none
 * in this harness by default: Xcelerit's own showcase benchmark suite
 * doesn't thread XAD's tape across paths either, so 1 thread is "as it
 * ships", not a handicap invented for this comparison.
 *
 * `threads=N` (N>1) adds a std::thread pool *on top* of XAD -- one tape per
 * thread (a tape is a plain object, not a global, so this is safe), paths
 * split into contiguous ranges, partial sums combined at the end. This is
 * explicitly NOT "as it ships": neither XAD nor its own official benchmark
 * ships this pool, so every threads>1 row is scaffolding this harness
 * built, on the same footing MatLogica's own ThreadPool(8) or NablaTensor's
 * own threads(n) are *not* on -- see the article for why the threads=1 rows
 * are the primary measurement and the threads=N rows are a separate,
 * clearly-labelled experiment.
 *
 * XAD-Codegen (the compiled, 2-5x-faster backend the same official
 * benchmark suite shows) is a separate, closed-source, sales-gated
 * component with no self-serve trial -- see the article for why it has no
 * row here either.
 *
 * Mirrors CrnLadder.java / matlogica_bench.py methodology: same market
 * (S=K=100, vol=20%, r=3%, T=1), same one-step European / 252-fixing
 * arithmetic Asian, same 7-spot ladder, same warm-up-then-3-ladders-of-
 * seeds-42-44 protocol, cold = first call of a ladder, warm = mean of the
 * other six, reported numbers are medians over the three ladders.
 *
 *   ./xad_ladder european|asian PATHS [greeks] [cached] [threads=N]
 */
#include <XAD/XAD.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <tuple>
#include <vector>

using Clock = std::chrono::steady_clock;

static bool IS_EUROPEAN;
static int STEPS;
static const double K0 = 100.0, VOL0 = 0.20, RATE0 = 0.03, T0 = 1.0;
static const double SPOTS[7] = {98, 99, 99.5, 100, 100.5, 101, 102};

// Single-path payoff, discounted. z points at STEPS standard-normal draws
// for this path. Templated so the exact same kernel runs as plain double
// (price-only, no tape at all) and as XAD's adjoint AD type (value+Greeks)
// -- one source of truth, no chance of the two passes silently pricing
// different products.
template <typename T>
T pathPayoff(const T& s, const T& k, const T& vol, const T& rate, const T& tExp,
             const double* z)
{
    using std::exp;
    using std::sqrt;
    T dt = tExp / double(STEPS);
    T sqrt_dt = sqrt(dt);
    T zero = s - s;
    T price = s;
    T running = zero;
    for (int i = 0; i < STEPS; ++i)
    {
        T diffusion = vol * sqrt_dt * z[i];
        price = price * exp((rate - T(0.5) * vol * vol) * dt + diffusion);
        running = running + price;
    }
    T avg = IS_EUROPEAN ? price : running / double(STEPS);
    T payoff = avg - k;
    T disc = exp(-rate * tExp);
    return disc * (payoff > zero ? payoff : zero);
}

struct CallOut
{
    double price = 0, delta = 0, dK = 0, vega = 0, rho = 0, dT = 0;
};

static void fillNormals(std::vector<double>& z, unsigned long long seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& v : z) v = nd(rng);
}

// Splits [0, paths) into nThreads contiguous ranges. threads=1 takes the
// same code path as every other single-threaded row in this harness (no
// std::thread object even constructed), so the threads=1 numbers are
// unaffected by this function existing.
static std::vector<std::pair<long, long>> splitRanges(long paths, int nThreads)
{
    std::vector<std::pair<long, long>> ranges;
    long chunk = (paths + nThreads - 1) / nThreads;
    for (int t = 0; t < nThreads; ++t)
    {
        long lo = std::min(paths, t * chunk);
        long hi = std::min(paths, lo + chunk);
        ranges.push_back({lo, hi});
    }
    return ranges;
}

static CallOut priceOnly(double spot, const std::vector<double>& z, long paths, int nThreads)
{
    auto work = [&](long lo, long hi) {
        double sum = 0.0;
        for (long p = lo; p < hi; ++p)
            sum += pathPayoff<double>(spot, K0, VOL0, RATE0, T0, &z[p * STEPS]);
        return sum;
    };
    if (nThreads <= 1) return {work(0, paths) / paths};

    auto ranges = splitRanges(paths, nThreads);
    std::vector<double> partial(nThreads, 0.0);
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t)
        pool.emplace_back([&, t] { partial[t] = work(ranges[t].first, ranges[t].second); });
    for (auto& th : pool) th.join();
    double sum = 0.0;
    for (double s : partial) sum += s;
    return {sum / paths};
}

// XAD's own per-path tape idiom (see file header): one tape recording per
// path, derivatives summed across paths and averaged, exactly like
// heston_xad.cpp. nThreads>1 gives each thread its own tape (a tape is a
// plain object; nothing in XAD requires it to be process-global) over its
// own path range -- the harness's own pool, not XAD's.
static CallOut priceWithGreeks(double spot, const std::vector<double>& z, long paths, int nThreads)
{
    using mode = xad::adj<double>;
    using AD = mode::active_type;

    struct Sums { double price = 0, d_s = 0, d_k = 0, d_vol = 0, d_rate = 0, d_t = 0; };

    auto work = [&](long lo, long hi) {
        mode::tape_type tape;
        Sums s;
        for (long p = lo; p < hi; ++p)
        {
            tape.clearAll();
            AD sp = spot, k = K0, vol = VOL0, rate = RATE0, tExp = T0;
            tape.registerInput(sp);
            tape.registerInput(k);
            tape.registerInput(vol);
            tape.registerInput(rate);
            tape.registerInput(tExp);
            tape.newRecording();

            AD payoff = pathPayoff<AD>(sp, k, vol, rate, tExp, &z[p * STEPS]);

            tape.registerOutput(payoff);
            derivative(payoff) = 1.0;
            tape.computeAdjoints();

            s.price += value(payoff);
            s.d_s += derivative(sp);
            s.d_k += derivative(k);
            s.d_vol += derivative(vol);
            s.d_rate += derivative(rate);
            s.d_t += derivative(tExp);
        }
        return s;
    };

    Sums total;
    if (nThreads <= 1)
    {
        total = work(0, paths);
    }
    else
    {
        auto ranges = splitRanges(paths, nThreads);
        std::vector<Sums> partial(nThreads);
        std::vector<std::thread> pool;
        for (int t = 0; t < nThreads; ++t)
            pool.emplace_back([&, t] { partial[t] = work(ranges[t].first, ranges[t].second); });
        for (auto& th : pool) th.join();
        for (auto& s : partial)
        {
            total.price += s.price;
            total.d_s += s.d_s;
            total.d_k += s.d_k;
            total.d_vol += s.d_vol;
            total.d_rate += s.d_rate;
            total.d_t += s.d_t;
        }
    }
    CallOut out;
    out.price = total.price / paths;
    out.delta = total.d_s / paths;
    out.dK = total.d_k / paths;
    out.vega = total.d_vol / paths;
    out.rho = total.d_rate / paths;
    out.dT = total.d_t / paths;
    return out;
}

static double peakRssMb()
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024.0; // ru_maxrss is KB on Linux
}

static double median3(double a, double b, double c)
{
    double v[3] = {a, b, c};
    std::sort(v, v + 3);
    return v[1];
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr,
                      "usage: %s european|asian PATHS [greeks] [cached] [threads=N]\n", argv[0]);
        return 1;
    }
    std::string product = argv[1];
    long paths = std::atol(argv[2]);
    bool greeks = false, cached = false;
    int nThreads = 1;
    for (int i = 3; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "greeks") == 0) greeks = true;
        if (std::strcmp(argv[i], "cached") == 0) cached = true;
        if (std::strncmp(argv[i], "threads=", 8) == 0) nThreads = std::atoi(argv[i] + 8);
    }
    IS_EUROPEAN = (product == "european");
    STEPS = IS_EUROPEAN ? 1 : 252;

    auto callOne = [&](double spot, std::vector<double>& z, unsigned long long seed,
                        bool regenNow) -> CallOut
    {
        if (regenNow) fillNormals(z, seed);
        return greeks ? priceWithGreeks(spot, z, paths, nThreads)
                       : priceOnly(spot, z, paths, nThreads);
    };

    auto runLadder = [&](unsigned long long seed, std::vector<CallOut>* firstLadderOut)
        -> std::pair<double, double>
    {
        std::vector<double> z(paths * STEPS);
        if (cached) fillNormals(z, seed); // generated once, reused for every spot below

        double cold = -1;
        std::vector<double> warm;
        for (int i = 0; i < 7; ++i)
        {
            auto t0 = Clock::now();
            CallOut r = callOne(SPOTS[i], z, seed, /*regenNow=*/!cached);
            auto t1 = Clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();
            if (i == 0)
                cold = secs;
            else
                warm.push_back(secs);
            if (firstLadderOut) firstLadderOut->push_back(r);
        }
        double warmMean = 0;
        for (double w : warm) warmMean += w;
        warmMean /= warm.size();
        return {cold, warmMean};
    };

    // Warm-up ladder, discarded, seed 7 -- lets CPU frequency scaling and
    // allocators settle before anything is timed, same as ladder.py.
    runLadder(7, nullptr);

    std::vector<CallOut> ladder0;
    auto [c1, w1] = runLadder(42, &ladder0);
    auto [c2, w2] = runLadder(43, nullptr);
    auto [c3, w3] = runLadder(44, nullptr);

    double coldSecs = median3(c1, c2, c3);
    double warmSecs = median3(w1, w2, w3);
    double coldMpaths = paths / coldSecs / 1e6;
    double warmMpaths = paths / warmSecs / 1e6;

    const CallOut& at100 = ladder0[3]; // SPOTS[3] == 100.0

    std::printf(
        "RESULT {\"lib\": \"XAD (C++, adj, per-path tape%s) %s\", \"paths\": %ld, "
        "\"pass\": \"%s\", \"cold\": %.6f, \"warm\": %.6f, "
        "\"at100\": {\"price\": %.10f",
        nThreads > 1 ? ", harness thread pool" : "", product.c_str(), paths,
        greeks ? "value+G" : "price", coldMpaths, warmMpaths, at100.price);
    if (greeks)
        std::printf(", \"delta\": %.10f, \"dK\": %.10f, \"vega\": %.10f, \"rho\": %.10f, "
                    "\"dT\": %.10f",
                    at100.delta, at100.dK, at100.vega, at100.rho, at100.dT);
    std::printf("}, \"threads\": %d, \"draws\": \"%s\"}\n", nThreads,
                cached ? "generated once per ladder, reused across spots"
                       : "regenerated every call");
    std::printf("peak RSS MB %.1f\n", peakRssMb());
    return 0;
}
