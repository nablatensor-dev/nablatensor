//! RustQuant: `MonteCarloPricer::price_monte_carlo` for `EuropeanVanillaOption` and
//! `AsianOption` on a `GeometricBrownianMotion`, Euler–Maruyama on the level (the only
//! scheme it has), rayon-parallel (`parallel: true`, RAYON_NUM_THREADS=8). There is no
//! seed argument — every path draws from `rand::thread_rng()` — so the ladder's seeds
//! are ignored, and the bump Greeks are on independent draws, as shipped.
//!
//!   RAYON_NUM_THREADS=8 cargo run --release -- european|asian PATHS [greeks]
use RustQuant::instruments::*;
use RustQuant::stochastics::*;
use std::time::Instant;
use time::macros::date;

const SPOTS: [f64; 7] = [98.0, 99.0, 99.5, 100.0, 100.5, 101.0, 102.0];

struct Out { price: f64, delta: f64, dk: f64, vega: f64, rho: f64, dt: f64 }

fn npv(product: &str, s: f64, k: f64, vol: f64, r: f64, t: f64, steps: usize, paths: usize) -> f64 {
    let gbm = GeometricBrownianMotion::new(r, vol);
    let cfg = StochasticProcessConfig::new(s, 0.0, t, steps, paths, true);
    let expiry = date!(2027 - 01 - 01);
    if product == "european" {
        EuropeanVanillaOption::new(k, expiry, TypeFlag::Call).price_monte_carlo(&gbm, &cfg, r)
    } else {
        let contract = OptionContractBuilder::default()
            .type_flag(TypeFlag::Call)
            .exercise_flag(ExerciseFlag::European { expiry })
            .strike_flag(Some(StrikeFlag::Fixed))
            .build().unwrap();
        AsianOption::new(contract, AveragingMethod::ArithmeticDiscrete, Some(k)).price_monte_carlo(&gbm, &cfg, r)
    }
}

fn value(product: &str, s: f64, steps: usize, paths: usize, greeks: bool) -> Out {
    let (k, vol, r, t) = (100.0, 0.20, 0.03, 1.0);
    let v = npv(product, s, k, vol, r, t, steps, paths);
    if !greeks { return Out { price: v, delta: 0.0, dk: 0.0, vega: 0.0, rho: 0.0, dt: 0.0 }; }
    let (hs, hk, hv, hr, ht) = (0.01, 0.01, 1e-4, 1e-4, 1.0 / 252.0);
    Out {
        price: v,
        delta: (npv(product, s + hs, k, vol, r, t, steps, paths) - v) / hs,
        dk: (npv(product, s, k + hk, vol, r, t, steps, paths) - v) / hk,
        vega: (npv(product, s, k, vol + hv, r, t, steps, paths) - v) / hv,
        rho: (npv(product, s, k, vol, r + hr, t, steps, paths) - v) / hr,
        dt: (npv(product, s, k, vol, r, t + ht, steps, paths) - v) / ht,
    }
}

fn median(v: &mut Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let product = a[1].as_str();
    let paths: usize = a[2].parse().unwrap();
    let greeks = a.len() > 3 && a[3] == "greeks";
    let steps = if product == "european" { 1 } else { 252 };
    let ladders: usize = std::env::var("LADDERS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let warmup: usize = std::env::var("WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    for _ in 0..warmup { value(product, 100.0, steps, paths, greeks); }
    let (mut cold, mut warm) = (vec![], vec![]);
    let mut at100 = None;
    let mut ladder0: Vec<String> = vec![];
    for l in 0..ladders {
        let mut rates = vec![];
        for &s in SPOTS.iter() {
            let t0 = Instant::now();
            let o = value(product, s, steps, paths, greeks);
            rates.push(paths as f64 / t0.elapsed().as_secs_f64() / 1e6);
            if l == 0 { ladder0.push(format!("{{\"price\": {}, \"delta\": {}, \"dK\": {}, \"vega\": {}, \"rho\": {}, \"dT\": {}}}", o.price, o.delta, o.dk, o.vega, o.rho, o.dt)); }
            if s == 100.0 { at100 = Some(o); }
        }
        cold.push(rates[0]);
        warm.push(rates[1..].iter().sum::<f64>() / 6.0);
        println!("  ladder {}: cold {:.3}  warm {:?}", l, rates[0], &rates[1..].iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    }
    let o = at100.unwrap();
    println!("RESULT {{\"lib\": \"RustQuant 0.3.1 {}\", \"paths\": {}, \"pass\": \"{}\", \"cold\": {:.4}, \"warm\": {:.4}, \"at100\": {{\"price\": {}, \"delta\": {}, \"dK\": {}, \"vega\": {}, \"rho\": {}, \"dT\": {}}}, \"threads\": {}, \"greeks\": \"{}\", \"draws\": \"thread_rng, no seed\", \"ladder0\": [{}]}}",
        product, paths, if greeks { "value+G" } else { "price" }, median(&mut cold), median(&mut warm),
        o.price, o.delta, o.dk, o.vega, o.rho, o.dt,
        rayon_threads(), if greeks { "bump x5, independent draws" } else { "-" }, ladder0.join(", "));
}

fn rayon_threads() -> usize { std::env::var("RAYON_NUM_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)) }
