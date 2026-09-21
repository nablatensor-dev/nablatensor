/*
 * Copyright 2026 The NablaTensor Authors
 * Licensed under the Apache License, Version 2.0.
 */
package com.nablatensor.quant;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.nablatensor.codegen.Of;
import com.nablatensor.engine.Nabla;
import org.junit.jupiter.api.Test;

final class BuilderMarketTest {

  @Test
  void builderRequiresPrimitivesAndCanCopy() {
    assertThrows(IllegalStateException.class, BuilderMarket::createWithoutRate);
    BuilderMarket original = market(2.0, 3.0, 4.0);
    BuilderMarket copy = BuilderMarket.of().from(original).spot(5.0).build();
    assertEquals(5.0, copy.spot());
    assertEquals(3.0, copy.vol());
    assertEquals(4.0, copy.rate());
  }

  @Test
  void builderMarketHasPriceAndGreekParity() {
    BuilderMarket market = market(2.0, 3.0, 4.0);
    try (Nabla.TypedPricer<BuilderMarket> pricer = Nabla.model(market, (rec, in) ->
        rec.output(in.of(BuilderMarket::spot).mul(in.of(BuilderMarket::vol))
            .add(in.of(BuilderMarket::rate))))
        .fp64().greeks().on("cpu-jit").build()) {
      Nabla.TypedValuation<BuilderMarket> value = pricer.value().with(market).run();
      assertEquals(10.0, value.price(), 0.0);
      assertEquals(3.0, value.greek(BuilderMarket::spot), 0.0);
      assertEquals(2.0, value.greek(BuilderMarket::vol), 0.0);
      assertEquals(1.0, value.greek(BuilderMarket::rate), 0.0);
    }
  }

  private static BuilderMarket market(double spot, double vol, double rate) {
    return BuilderMarket.of().spot(spot).vol(vol).rate(rate).build();
  }
}

@Of
final class BuilderMarket {
  private final double spot;
  private final double vol;
  private final double rate;

  private BuilderMarket(double spot, double vol, double rate) {
    this.spot = spot;
    this.vol = vol;
    this.rate = rate;
  }

  static BuilderMarket create(double spot, double vol, double rate) {
    return new BuilderMarket(spot, vol, rate);
  }

  static BuilderMarketBuilder of() {
    return new BuilderMarketBuilder();
  }

  static BuilderMarket createWithoutRate() {
    return of().spot(1.0).vol(2.0).build();
  }

  public double spot() { return spot; }
  public double vol() { return vol; }
  public double rate() { return rate; }
}
