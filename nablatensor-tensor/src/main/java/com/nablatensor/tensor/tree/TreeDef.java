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
package com.nablatensor.tensor.tree;

import java.util.List;
import java.util.Objects;
import java.util.function.UnaryOperator;

/** An immutable description of a pytree's container structure. */
public final class TreeDef {

  /** The kind of node described by this definition. */
  public enum Kind {
    LEAF, NULL, LIST, MAP, ARRAY, RECORD, CUSTOM
  }

  private final Kind kind;
  private final Class<?> type;
  private final Object metadata;
  private final List<TreeDef> children;
  private final int leafCount;

  TreeDef(Kind kind, Class<?> type, Object metadata, List<TreeDef> children) {
    this.kind = Objects.requireNonNull(kind, "kind");
    this.type = type;
    this.metadata = metadata;
    this.children = List.copyOf(children);
    this.leafCount = kind == Kind.LEAF
        ? 1
        : this.children.stream().mapToInt(TreeDef::leafCount).sum();
  }

  static TreeDef leaf() {
    return new TreeDef(Kind.LEAF, null, null, List.of());
  }

  static TreeDef node(Kind kind, Class<?> type, Object metadata, List<TreeDef> children) {
    return new TreeDef(kind, type, metadata, children);
  }

  /** Node category. */
  public Kind kind() {
    return kind;
  }

  /**
   * The record/custom class, or an array component type; {@code null}
   * for kinds which do not need a Java type to rebuild.
   */
  public Class<?> type() {
    return type;
  }

  /** Node-specific immutable metadata, such as sorted map keys. */
  public Object metadata() {
    return metadata;
  }

  /** Child definitions in stable traversal order. */
  public List<TreeDef> children() {
    return children;
  }

  /** Number of leaf values required by {@link TreeUtil#unflatten(TreeDef, List)}. */
  public int leafCount() {
    return leafCount;
  }

  TreeDef replaceLeaves(TreeDef replacement) {
    if (kind == Kind.LEAF) {
      return replacement;
    }

    if (children.isEmpty()) {
      return this;
    }
    return new TreeDef(kind, type, metadata,
        children.stream().map(child -> child.replaceLeaves(replacement)).toList());
  }

  /** Returns this structure with every array component type transformed by {@code mapper}. */
  public TreeDef mapArrayTypes(UnaryOperator<Class<?>> mapper) {
    Objects.requireNonNull(mapper, "mapper");
    Class<?> mappedType = kind == Kind.ARRAY ? mapper.apply(type) : type;
    return new TreeDef(kind, mappedType, metadata,
        children.stream().map(child -> child.mapArrayTypes(mapper)).toList());
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof TreeDef that)) {
      return false;
    }
    return kind == that.kind
        && Objects.equals(type, that.type)
        && Objects.equals(metadata, that.metadata)
        && children.equals(that.children);
  }

  @Override
  public int hashCode() {
    return Objects.hash(kind, type, metadata, children);
  }

  @Override
  public String toString() {
    return "TreeDef[kind=" + kind + ", children=" + children.size()
        + ", leafCount=" + leafCount + "]";
  }
}
