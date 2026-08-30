/*
 * Copyright 2026 The NablaTensor Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nablatensor.quant;

import com.nablatensor.engine.AadRecorder;
import com.nablatensor.engine.SDouble;
import com.nablatensor.engine.Nabla;
import com.nablatensor.ops.Smooth;
import java.util.ArrayList;
import java.util.List;

/**
 * Variance-reduction hooks that live entirely in the recorded payoff (Seam 4):
 * they wrap a {@link PathPayoff} — a payoff written against an explicit
 * {@link Draws} source — and re-record it with a transformed draw stream. No
 * engine change; the adjoint flows through the wrapped graph as usual.
 *
 * <p>The result is a {@link Product}, so it drops straight into
 * {@link MonteCarlo}.
 */
public final class Hooks {

  /** A source of per-step standard-normal draws for a {@link PathPayoff}. */
  @FunctionalInterface
  public interface Draws {
    SDouble next();
  }

  /** A payoff written so its randomness comes only from an injected {@link Draws}. */
  @FunctionalInterface
  public interface PathPayoff {
    SDouble value(AadRecorder rec, Nabla.Inputs<EquityMarket> in, Draws draws, TimeGrid grid);
  }

  private Hooks() {
  }

  /**
   * Antithetic pairing: records the payoff twice on the same tape, once with the
   * draw stream {@code z} and once with {@code -z}, and averages. Unbiased;
   * reduces variance for payoffs with an odd component in {@code z}.
   */
  public static Product<EquityMarket> antithetic(PathPayoff payoff) {
    return new Named("antithetic", (rec, in, grid) -> {
      List<SDouble> z = new ArrayList<>();
      SDouble a = payoff.value(rec, in, capturing(rec, z), grid);
      SDouble b = payoff.value(rec, in, replayNegated(z), grid);
      rec.output(a.add(b).mul(0.5));
    });
  }

  /**
   * Control variate: {@code Y* = Y - beta (X - E[X])}, with the control {@code X}
   * driven by the <em>same</em> draws as the target {@code Y}. Unbiased for any
   * {@code beta}; {@code beta = 1} with a well-correlated control is the usual
   * choice. The control must not consume more draws per step than the target.
   *
   * @param target      the payoff being priced
   * @param control     a payoff with a known analytic mean on the same path
   * @param controlMean {@code E[X]}, the analytic value of the control
   * @param beta        control coefficient
   */
  public static Product<EquityMarket> controlVariate(PathPayoff target, PathPayoff control,
                                       double controlMean, double beta) {
    return new Named("control-variate", (rec, in, grid) -> {
      List<SDouble> z = new ArrayList<>();
      SDouble y = target.value(rec, in, capturing(rec, z), grid);
      SDouble x = control.value(rec, in, replay(z), grid);
      rec.output(y.sub(x.sub(controlMean).mul(beta)));
    });
  }

  /**
   * Importance sampling: draw {@code z ~ N(0,1)}, feed the payoff the drifted
   * stream {@code z + muPerStep}, and multiply by the per-path likelihood ratio
   * {@code exp(-muPerStep * sum z - 0.5 muPerStep^2 n)}. Unbiased for any
   * {@code muPerStep}; a positive shift concentrates paths where an
   * out-of-the-money payoff pays.
   */
  public static Product<EquityMarket> importanceSampling(PathPayoff payoff, double muPerStep) {
    return new Named("importance-sampling", (rec, in, grid) -> {
      SDouble[] logW = {rec.constant(0.0)};
      int[] n = {0};
      Draws drifted = () -> {
        SDouble z = rec.randn();
        logW[0] = logW[0].add(z.mul(-muPerStep));
        n[0]++;
        return z.add(muPerStep);
      };
      SDouble value = payoff.value(rec, in, drifted, grid);
      SDouble weight = logW[0].add(-0.5 * muPerStep * muPerStep * n[0]).exp();
      rec.output(value.mul(weight));
    });
  }

  /**
   * Path filter: computes {@code E[payoff * 1{condition > 0}]} with the indicator
   * smoothed at {@code width}. The condition is a second payoff over the same
   * draws whose sign selects the paths to keep. Note this is
   * {@code E[f * 1{...}]}, not the conditional expectation {@code E[f | ...]}.
   */
  public static Product<EquityMarket> pathFilter(PathPayoff payoff, PathPayoff condition, double width) {
    return new Named("path-filter", (rec, in, grid) -> {
      List<SDouble> z = new ArrayList<>();
      SDouble f = payoff.value(rec, in, capturing(rec, z), grid);
      SDouble c = condition.value(rec, in, replay(z), grid);
      rec.output(f.mul(Smooth.step(rec, c, width)));
    });
  }

  private static Draws capturing(AadRecorder rec, List<SDouble> sink) {
    return () -> {
      SDouble draw = rec.randn();
      sink.add(draw);
      return draw;
    };
  }

  private static Draws replay(List<SDouble> source) {
    int[] i = {0};
    return () -> source.get(i[0]++);
  }

  private static Draws replayNegated(List<SDouble> source) {
    int[] i = {0};
    return () -> source.get(i[0]++).neg();
  }

  private record Named(String label, Product<EquityMarket> body) implements Product<EquityMarket> {
    @Override
    public void record(AadRecorder rec, Nabla.Inputs<EquityMarket> in, TimeGrid grid) {
      body.record(rec, in, grid);
    }
  }
}
