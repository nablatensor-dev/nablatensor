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
package com.nablatensor.annotation;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a public type, member or package that is <em>not</em> part of the
 * supported API. It is public only because the module layout requires it to be
 * reachable from another {@code nablatensor-*} module; it may change or be
 * removed in any release, with no deprecation cycle.
 *
 * <p>Application code should depend only on unannotated public types in
 * non-{@code internal} packages. See {@code docs/api-stability.md}.
 */
@Documented
@Retention(RetentionPolicy.CLASS)
@Target({
    ElementType.TYPE,
    ElementType.METHOD,
    ElementType.CONSTRUCTOR,
    ElementType.FIELD,
    ElementType.PACKAGE
})
public @interface Internal {
}
