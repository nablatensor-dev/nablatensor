package com.nablatensor.codegen;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/** Generates a named, fluent draft for an immutable configuration class. */
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.SOURCE)
public @interface Of {}
