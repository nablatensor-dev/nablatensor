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
package com.nablatensor.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.nablatensor.tensor.spi.ComputeBackend;
import com.nablatensor.tensor.spi.DeviceBuffer;
import com.nablatensor.tensor.tree.TreeUtil;

import java.lang.reflect.Proxy;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

class TreeCloseAllTest {

  @Test
  void closesRepeatedTensorIdentityOnlyOnce() {
    AtomicInteger releases = new AtomicInteger();
    ComputeBackend backend = (ComputeBackend) Proxy.newProxyInstance(
        ComputeBackend.class.getClassLoader(),
        new Class<?>[] {ComputeBackend.class},
        (proxy, method, args) -> {
          if (method.getName().equals("release")) {
            releases.incrementAndGet();
          }
          return null;
        });
    DeviceBuffer buffer = (DeviceBuffer) Proxy.newProxyInstance(
        DeviceBuffer.class.getClassLoader(),
        new Class<?>[] {DeviceBuffer.class},
        (proxy, method, args) -> null);
    Tensor tensor = new Tensor(backend, buffer);

    TreeUtil.closeAll(List.of(tensor, List.of(tensor)));

    assertEquals(1, releases.get());
  }
}
