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

import com.nablatensor.tensor.Device;
import com.nablatensor.tensor.NablaTensors;
import com.nablatensor.tensor.Shape;
import com.nablatensor.tensor.Tensor;

import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.RecordComponent;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.UnaryOperator;

/** Utilities for traversing and rebuilding nested parameter trees. */
public final class TreeUtil {

  /** A flattened tree and the immutable definition needed to rebuild it. */
  public record Flattened(List<Object> leaves, TreeDef def) {
    public Flattened {
      leaves = Collections.unmodifiableList(new ArrayList<>(leaves));
      Objects.requireNonNull(def, "def");
    }
  }

  /** Children and auxiliary metadata returned by a registered node flattener. */
  public record Node(List<Object> children, Object metadata) {
    public Node {
      children = Collections.unmodifiableList(new ArrayList<>(children));
    }
  }

  /** A flat tensor and a function that rebuilds the original tensor tree. */
  public record Ravelled(Tensor flat, Function<Tensor, Object> unravel) {
    public Ravelled {
      Objects.requireNonNull(flat, "flat");
      Objects.requireNonNull(unravel, "unravel");
    }
  }

  /** Decomposes a registered node into children and auxiliary metadata. */
  @FunctionalInterface
  public interface Flattener {
    Node flatten(Object tree);
  }

  /** Rebuilds a registered node from its auxiliary metadata and rebuilt children. */
  @FunctionalInterface
  public interface Unflattener {
    Object unflatten(Object metadata, List<Object> children);
  }

  /** Maps an arbitrary number of leaf values at the same tree position. */
  @FunctionalInterface
  public interface NaryOperator {
    Object apply(List<Object> values);
  }

  private record Registration(Flattener flattener, Unflattener unflattener) {
  }

  private static final Map<Class<?>, Registration> REGISTRATIONS = new ConcurrentHashMap<>();

  private TreeUtil() {
  }

  /** Flattens all non-container values into leaves. */
  public static Flattened flatten(Object tree) {
    return flatten(tree, ignored -> false);
  }

  /**
   * Flattens a tree with an additional leaf predicate. {@code null} remains an
   * empty node even when the predicate accepts it.
   */
  public static Flattened flatten(Object tree, Predicate<Object> isLeaf) {
    Objects.requireNonNull(isLeaf, "isLeaf");
    List<Object> leaves = new ArrayList<>();
    return new Flattened(leaves, flattenInto(tree, isLeaf, leaves));
  }

  /** Rebuilds a tree, requiring exactly {@link TreeDef#leafCount()} leaves. */
  public static Object unflatten(TreeDef def, List<?> leaves) {
    Objects.requireNonNull(def, "def");
    Objects.requireNonNull(leaves, "leaves");
    if (leaves.size() != def.leafCount()) {
      throw new IllegalArgumentException(
          "TreeDef requires " + def.leafCount() + " leaves but received " + leaves.size());
    }
    return build(def, leaves, new int[] {0});
  }

  /** Returns a tree's leaves in deterministic traversal order. */
  public static List<Object> leaves(Object tree) {
    return flatten(tree).leaves();
  }

  /** Returns a tree's immutable structural definition. */
  public static TreeDef structure(Object tree) {
    return flatten(tree).def();
  }

  /** Applies a unary operation to every leaf and rebuilds the original structure. */
  public static <T> Object map(UnaryOperator<T> fn, Object tree) {
    Objects.requireNonNull(fn, "fn");
    Flattened flattened = flatten(tree);
    List<Object> mapped = new ArrayList<>(flattened.leaves().size());
    for (Object leaf : flattened.leaves()) {
      @SuppressWarnings("unchecked")
      T typedLeaf = (T) leaf;
      mapped.add(fn.apply(typedLeaf));
    }
    return unflatten(flattened.def(), mapped);
  }

  /** Applies a binary operation to matching leaves in two trees. */
  public static <A, B, R> Object map(
      BiFunction<? super A, ? super B, ? extends R> fn, Object first, Object second) {
    Objects.requireNonNull(fn, "fn");
    Flattened left = flatten(first);
    Flattened right = flatten(second);
    requireSameStructure(left.def(), right.def());
    List<Object> mapped = new ArrayList<>(left.leaves().size());
    for (int i = 0; i < left.leaves().size(); i++) {
      @SuppressWarnings("unchecked")
      A a = (A) left.leaves().get(i);
      @SuppressWarnings("unchecked")
      B b = (B) right.leaves().get(i);
      mapped.add(fn.apply(a, b));
    }
    return unflatten(left.def(), mapped);
  }

  /** Applies an operation to matching leaves in one or more trees. */
  public static Object map(NaryOperator fn, Object first, Object... rest) {
    Objects.requireNonNull(fn, "fn");
    Flattened base = flatten(first);
    List<Flattened> flattenedRest = new ArrayList<>(rest.length);
    for (Object tree : rest) {
      Flattened flattened = flatten(tree);
      requireSameStructure(base.def(), flattened.def());
      flattenedRest.add(flattened);
    }

    List<Object> mapped = new ArrayList<>(base.leaves().size());
    for (int i = 0; i < base.leaves().size(); i++) {
      List<Object> values = new ArrayList<>(flattenedRest.size() + 1);
      values.add(base.leaves().get(i));
      for (Flattened flattened : flattenedRest) {
        values.add(flattened.leaves().get(i));
      }
      mapped.add(fn.apply(Collections.unmodifiableList(values)));
    }
    return unflatten(base.def(), mapped);
  }

  /** Folds leaves from left to right, beginning with {@code initial}. */
  public static <T> T reduce(BinaryOperator<T> fn, Object tree, T initial) {
    Objects.requireNonNull(fn, "fn");
    T result = initial;
    for (Object leaf : leaves(tree)) {
      @SuppressWarnings("unchecked")
      T typedLeaf = (T) leaf;
      result = fn.apply(result, typedLeaf);
    }
    return result;
  }

  /** Swaps two nested tree levels, as in JAX's {@code tree_transpose}. */
  public static Object transpose(TreeDef outer, TreeDef inner, Object tree) {
    Objects.requireNonNull(outer, "outer");
    Objects.requireNonNull(inner, "inner");
    TreeDef expected = outer.replaceLeaves(inner);
    Flattened flattened = flatten(tree);
    if (!expected.equals(flattened.def())) {
      throw new IllegalArgumentException("tree structure does not match outer composed with inner");
    }

    int innerLeafCount = inner.leafCount();
    if (innerLeafCount == 0) {
      return unflatten(inner, List.of());
    }

    List<Object> transposedChildren = new ArrayList<>(innerLeafCount);
    for (int innerIndex = 0; innerIndex < innerLeafCount; innerIndex++) {
      List<Object> outerLeaves = new ArrayList<>(outer.leafCount());
      for (int offset = innerIndex; offset < flattened.leaves().size(); offset += innerLeafCount) {
        outerLeaves.add(flattened.leaves().get(offset));
      }
      transposedChildren.add(unflatten(outer, outerLeaves));
    }
    return unflatten(inner, transposedChildren);
  }

  /**
   * Registers a node type. Registration is exact-class based, matching JAX's
   * node registration semantics.
   */
  public static void register(Class<?> type, Flattener flattener, Unflattener unflattener) {
    REGISTRATIONS.put(Objects.requireNonNull(type, "type"),
        new Registration(Objects.requireNonNull(flattener, "flattener"),
            Objects.requireNonNull(unflattener, "unflattener")));
  }

  /**
   * Closes each distinct tensor leaf once. Identity, rather than
   * {@code equals}, is intentional: a repeated tensor is one owned buffer.
   */
  public static void closeAll(Object tree) {
    Set<Tensor> seen = Collections.newSetFromMap(new IdentityHashMap<>());
    for (Object leaf : leaves(tree)) {
      if (leaf instanceof Tensor tensor && seen.add(tensor)) {
        tensor.close();
      }
    }
  }

  /**
   * Flattens a Tensor-leaf tree into one rank-1 Tensor. Until concat and slicing
   * primitives exist, this operation intentionally transfers through host memory.
   */
  public static Ravelled ravel(Object tree) {
    Flattened flattened = flatten(tree);
    List<Tensor> tensors = new ArrayList<>(flattened.leaves().size());
    long totalSize = 0;
    for (Object leaf : flattened.leaves()) {
      if (!(leaf instanceof Tensor tensor)) {
        throw new IllegalArgumentException(
            "ravel requires Tensor leaves, found " + leaf.getClass().getName());
      }
      tensors.add(tensor);
      totalSize = Math.addExact(totalSize, tensor.shape().size());
    }

    Device device = tensors.isEmpty() ? NablaTensors.defaultDevice() : tensors.getFirst().device();
    for (Tensor tensor : tensors) {
      if (!tensor.device().equals(device)) {
        throw new IllegalArgumentException("ravel requires all Tensor leaves on one device");
      }
    }

    float[] values = new float[Math.toIntExact(totalSize)];
    List<Shape> shapes = new ArrayList<>(tensors.size());
    int offset = 0;
    for (Tensor tensor : tensors) {
      float[] leafValues = tensor.toFloatArray();
      System.arraycopy(leafValues, 0, values, offset, leafValues.length);
      offset += leafValues.length;
      shapes.add(tensor.shape());
    }
    Tensor flat = NablaTensors.arrayOn(values, Shape.of(values.length), device);
    TreeDef def = flattened.def();
    int expectedLength = values.length;
    return new Ravelled(flat, flatTensor -> {
      if (flatTensor.shape().size() != expectedLength) {
        throw new IllegalArgumentException(
            "unravel expected " + expectedLength + " values, got " + flatTensor.shape());
      }
      float[] flatValues = flatTensor.toFloatArray();
      List<Object> rebuiltLeaves = new ArrayList<>(shapes.size());
      int leafOffset = 0;
      try {
        for (Shape shape : shapes) {
          int size = Math.toIntExact(shape.size());
          float[] leafValues = java.util.Arrays.copyOfRange(
              flatValues, leafOffset, leafOffset + size);
          rebuiltLeaves.add(NablaTensors.arrayOn(leafValues, shape, flatTensor.device()));
          leafOffset += size;
        }
        return unflatten(def, rebuiltLeaves);
      } catch (RuntimeException failure) {
        closeAll(rebuiltLeaves);
        throw failure;
      }
    });
  }

  private static TreeDef flattenInto(Object tree, Predicate<Object> isLeaf, List<Object> leaves) {
    if (tree == null) {
      return TreeDef.node(TreeDef.Kind.NULL, null, null, List.of());
    }
    if (isLeaf.test(tree)) {
      leaves.add(tree);
      return TreeDef.leaf();
    }

    Registration registration = REGISTRATIONS.get(tree.getClass());
    if (registration != null) {
      Node node = Objects.requireNonNull(registration.flattener().flatten(tree),
          "registered flattener result");
      return TreeDef.node(TreeDef.Kind.CUSTOM, tree.getClass(), node.metadata(),
          flattenChildren(node.children(), isLeaf, leaves));
    }
    if (tree instanceof List<?> list) {
      return TreeDef.node(TreeDef.Kind.LIST, null, null, flattenChildren(list, isLeaf, leaves));
    }
    if (tree instanceof Map<?, ?> map) {
      List<Object> keys = sortedKeys(map);
      List<TreeDef> children = new ArrayList<>(keys.size());
      for (Object key : keys) {
        children.add(flattenInto(map.get(key), isLeaf, leaves));
      }
      return TreeDef.node(TreeDef.Kind.MAP, null, List.copyOf(keys), children);
    }
    Class<?> type = tree.getClass();
    if (type.isArray()) {
      int length = Array.getLength(tree);
      List<TreeDef> children = new ArrayList<>(length);
      for (int i = 0; i < length; i++) {
        children.add(flattenInto(Array.get(tree, i), isLeaf, leaves));
      }
      return TreeDef.node(TreeDef.Kind.ARRAY, type.getComponentType(), null, children);
    }
    if (type.isRecord()) {
      RecordComponent[] components = type.getRecordComponents();
      List<TreeDef> children = new ArrayList<>(components.length);
      for (RecordComponent component : components) {
        children.add(flattenInto(readComponent(tree, component), isLeaf, leaves));
      }
      return TreeDef.node(TreeDef.Kind.RECORD, type, null, children);
    }

    leaves.add(tree);
    return TreeDef.leaf();
  }

  private static List<TreeDef> flattenChildren(
      List<?> values, Predicate<Object> isLeaf, List<Object> leaves) {
    List<TreeDef> children = new ArrayList<>(values.size());
    for (Object value : values) {
      children.add(flattenInto(value, isLeaf, leaves));
    }
    return children;
  }

  private static Object build(TreeDef def, List<?> leaves, int[] index) {
    return switch (def.kind()) {
      case LEAF -> leaves.get(index[0]++);
      case NULL -> null;
      case LIST -> buildList(def.children(), leaves, index);
      case MAP -> buildMap(def, leaves, index);
      case ARRAY -> buildArray(def, leaves, index);
      case RECORD -> buildRecord(def, leaves, index);
      case CUSTOM -> buildCustom(def, leaves, index);
    };
  }

  private static List<Object> buildChildren(List<TreeDef> defs, List<?> leaves, int[] index) {
    List<Object> children = new ArrayList<>(defs.size());
    for (TreeDef child : defs) {
      children.add(build(child, leaves, index));
    }
    return children;
  }

  private static List<Object> buildList(List<TreeDef> defs, List<?> leaves, int[] index) {
    return buildChildren(defs, leaves, index);
  }

  private static Map<Object, Object> buildMap(TreeDef def, List<?> leaves, int[] index) {
    @SuppressWarnings("unchecked")
    List<Object> keys = (List<Object>) def.metadata();
    List<Object> values = buildChildren(def.children(), leaves, index);
    Map<Object, Object> result = new LinkedHashMap<>();
    for (int i = 0; i < keys.size(); i++) {
      result.put(keys.get(i), values.get(i));
    }
    return result;
  }

  private static Object buildArray(TreeDef def, List<?> leaves, int[] index) {
    Object array = Array.newInstance(def.type(), def.children().size());
    List<Object> values = buildChildren(def.children(), leaves, index);
    try {
      for (int i = 0; i < values.size(); i++) {
        Array.set(array, i, values.get(i));
      }
    } catch (IllegalArgumentException e) {
      throw new IllegalArgumentException(
          "cannot rebuild " + def.type().getTypeName() + "[] with replacement leaf values", e);
    }
    return array;
  }

  private static Object buildRecord(TreeDef def, List<?> leaves, int[] index) {
    Class<?> type = def.type();
    RecordComponent[] components = type.getRecordComponents();
    Class<?>[] parameterTypes = new Class<?>[components.length];
    for (int i = 0; i < components.length; i++) {
      parameterTypes[i] = components[i].getType();
    }
    try {
      Constructor<?> constructor = type.getDeclaredConstructor(parameterTypes);
      if (!constructor.canAccess(null) && !constructor.trySetAccessible()) {
        throw new IllegalArgumentException("record canonical constructor is not accessible: " + type.getName());
      }
      return constructor.newInstance(buildChildren(def.children(), leaves, index).toArray());
    } catch (NoSuchMethodException | InstantiationException | IllegalAccessException
             | InvocationTargetException e) {
      throw new IllegalArgumentException("cannot rebuild record " + type.getName(), e);
    }
  }

  private static Object buildCustom(TreeDef def, List<?> leaves, int[] index) {
    Registration registration = REGISTRATIONS.get(def.type());
    if (registration == null) {
      throw new IllegalStateException("custom pytree type is no longer registered: " + def.type().getName());
    }
    return registration.unflattener().unflatten(def.metadata(),
        Collections.unmodifiableList(buildChildren(def.children(), leaves, index)));
  }

  private static Object readComponent(Object record, RecordComponent component) {
    Method accessor = component.getAccessor();
    try {
      if (!accessor.canAccess(record) && !accessor.trySetAccessible()) {
        throw new IllegalArgumentException("record component is not accessible: "
            + record.getClass().getName() + "." + component.getName());
      }
      return accessor.invoke(record);
    } catch (IllegalAccessException | InvocationTargetException e) {
      throw new IllegalArgumentException("cannot read record component "
          + record.getClass().getName() + "." + component.getName(), e);
    }
  }

  private static List<Object> sortedKeys(Map<?, ?> map) {
    List<Object> keys = new ArrayList<>(map.keySet());
    for (Object key : keys) {
      if (!(key instanceof Comparable<?>)) {
        throw new IllegalArgumentException(
            "map key is not Comparable and cannot be deterministically flattened: " + key);
      }
    }
    try {
      keys.sort(TreeUtil::compareKeys);
      for (int i = 1; i < keys.size(); i++) {
        if (compareKeys(keys.get(i - 1), keys.get(i)) == 0
            && !keys.get(i - 1).equals(keys.get(i))) {
          throw new IllegalArgumentException(
              "map keys compare equal but are not equal: " + keys.get(i - 1) + ", " + keys.get(i));
        }
      }
      return keys;
    } catch (ClassCastException e) {
      throw new IllegalArgumentException("map keys must be mutually Comparable for deterministic flattening", e);
    }
  }

  @SuppressWarnings("unchecked")
  private static int compareKeys(Object left, Object right) {
    if (!(left instanceof Comparable<?> comparable)) {
      throw new IllegalArgumentException(
          "map key is not Comparable and cannot be deterministically flattened: " + left);
    }
    return ((Comparable<Object>) comparable).compareTo(right);
  }

  private static void requireSameStructure(TreeDef expected, TreeDef actual) {
    if (!expected.equals(actual)) {
      throw new IllegalArgumentException(
          "tree structures differ: expected " + expected + " but received " + actual);
    }
  }
}
