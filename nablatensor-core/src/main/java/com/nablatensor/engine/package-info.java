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

/**
 * The recording engine and its backend SPI.
 *
 * <p><b>API</b> — {@link com.nablatensor.engine.Nabla} (entry point),
 * {@link com.nablatensor.engine.SDouble}, {@link com.nablatensor.engine.AadRecorder},
 * {@link com.nablatensor.engine.AadTape}, {@link com.nablatensor.engine.AadResult},
 * {@link com.nablatensor.engine.AadOptions}, {@link com.nablatensor.engine.JitOptimizations}.
 *
 * <p><b>SPI</b> — implement {@link com.nablatensor.engine.AadEngine} and
 * {@link com.nablatensor.engine.AadExecutable} to add a backend; discovery is by
 * {@link java.util.ServiceLoader}. {@link com.nablatensor.engine.AadEngines}
 * selects among them.
 *
 * <p>Types annotated {@link com.nablatensor.annotation.Internal} (e.g.
 * {@code AbstractAadExecutable}, {@code CudaAadCodegen}, {@code AadCheckpointPlan})
 * are public only for cross-module reach and may change in any release. See
 * {@code docs/api-stability.md}.
 */
package com.nablatensor.engine;
