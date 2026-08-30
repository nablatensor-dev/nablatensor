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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class TreeUtilTest {

  record Pair<T>(T left, T right) {
  }

  private static final class Box {
    private final String name;
    private final Object value;

    private Box(String name, Object value) {
      this.name = name;
      this.value = value;
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof Box box && name.equals(box.name) && value.equals(box.value);
    }
  }

  @Test
  void flattensMapsInSortedKeyOrderAndRoundTripsContainers() {
    Map<String, Object> first = new HashMap<>();
    first.put("z", new Object[] {"three", null});
    first.put("a", new Pair<>("one", List.of("two")));
    Map<String, Object> second = new HashMap<>();
    second.put("a", new Pair<>("one", List.of("two")));
    second.put("z", new Object[] {"three", null});

    TreeUtil.Flattened flattened = TreeUtil.flatten(first);

    assertEquals(List.of("one", "two", "three"), flattened.leaves());
    assertEquals(flattened, TreeUtil.flatten(second));
    assertEquals(3, flattened.def().leafCount());

    @SuppressWarnings("unchecked")
    Map<String, Object> rebuilt = (Map<String, Object>) TreeUtil.unflatten(flattened.def(), flattened.leaves());
    assertEquals(first.get("a"), rebuilt.get("a"));
    assertArrayEquals((Object[]) first.get("z"), (Object[]) rebuilt.get("z"));
  }

  @Test
  void roundTripsListsObjectArraysRecordsAndNull() {
    Object tree = Arrays.asList(
        new Pair<>("left", new Object[] {"array", null}),
        null,
        List.of("tail"));

    TreeUtil.Flattened flattened = TreeUtil.flatten(tree);

    assertEquals(List.of("left", "array", "tail"), flattened.leaves());
    assertEquals(3, flattened.def().leafCount());
    @SuppressWarnings("unchecked")
    List<Object> rebuilt = (List<Object>) TreeUtil.unflatten(flattened.def(), flattened.leaves());
    assertEquals(null, rebuilt.get(1));
    assertEquals(List.of("tail"), rebuilt.get(2));
    Pair<?> rebuiltPair = (Pair<?>) rebuilt.get(0);
    assertEquals("left", rebuiltPair.left());
    assertArrayEquals(new Object[] {"array", null}, (Object[]) rebuiltPair.right());
  }

  @Test
  void primitiveArraysAreContainersAndPreserveTheirRuntimeType() {
    int[] values = {1, 2, 3};

    TreeUtil.Flattened flattened = TreeUtil.flatten(values);

    assertEquals(List.of(1, 2, 3), flattened.leaves());
    assertEquals(int.class, flattened.def().type());
    assertArrayEquals(values, (int[]) TreeUtil.unflatten(flattened.def(), flattened.leaves()));
    assertNotEquals(flattened.def(), TreeUtil.structure(new Integer[] {1, 2, 3}));
    assertArrayEquals(new int[] {2, 4, 6},
        (int[]) TreeUtil.map((Integer value) -> value * 2, values));
    assertThrows(IllegalArgumentException.class,
        () -> TreeUtil.map(Integer::sum, values, new Integer[] {1, 2, 3}));
    IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
        () -> TreeUtil.unflatten(flattened.def(), List.of("1", "2", "3")));
    assertEquals("cannot rebuild int[] with replacement leaf values", failure.getMessage());
  }

  @Test
  void treeDefinitionsAreStructuralAndMapsRequireComparableKeys() {
    TreeDef first = TreeUtil.structure(List.of(new Pair<>("a", "b")));
    TreeDef second = TreeUtil.structure(List.of(new Pair<>("x", "y")));

    assertEquals(first, second);
    assertEquals(first.hashCode(), second.hashCode());
    assertNotEquals(first, TreeUtil.structure(List.of(new Pair<>("a", null))));
    assertThrows(IllegalArgumentException.class,
        () -> TreeUtil.flatten(Map.of(new Object(), "value")));
    assertThrows(IllegalArgumentException.class,
        () -> TreeUtil.flatten(Map.of(1, "one", "2", "two")));
    assertThrows(IllegalArgumentException.class,
        () -> TreeUtil.unflatten(first, List.of("only-one-leaf")));
  }

  @Test
  void mapsReducesAndTransposesMatchingStructures() {
    Object mapped = TreeUtil.map((String value) -> value + "!", List.of("a", "b"));
    assertEquals(List.of("a!", "b!"), mapped);
    assertEquals("abc", TreeUtil.reduce(String::concat, List.of("a", "b", "c"), ""));

    Map<String, Integer> one = Map.of("a", 1, "b", 2);
    Map<String, Integer> two = Map.of("a", 10, "b", 20);
    @SuppressWarnings("unchecked")
    Map<String, Integer> summed = (Map<String, Integer>) TreeUtil.map(Integer::sum, one, two);
    assertEquals(Map.of("a", 11, "b", 22), summed);

    @SuppressWarnings("unchecked")
    Map<String, Integer> nary = (Map<String, Integer>) TreeUtil.map(
        values -> values.stream().mapToInt(value -> (Integer) value).sum(), one, two, one);
    assertEquals(Map.of("a", 12, "b", 24), nary);
    assertThrows(IllegalArgumentException.class,
        () -> TreeUtil.map(Integer::sum, one, List.of(1, 2)));

    Object transposed = TreeUtil.transpose(
        TreeUtil.structure(List.of(0, 0)),
        TreeUtil.structure(Map.of("a", 0, "b", 0)),
        List.of(Map.of("a", 1, "b", 2), Map.of("a", 3, "b", 4)));
    assertEquals(Map.of("a", List.of(1, 3), "b", List.of(2, 4)), transposed);
    assertThrows(IllegalArgumentException.class,
        () -> TreeUtil.transpose(TreeUtil.structure(List.of(0)), TreeUtil.structure(List.of(0)),
            List.of(1, 2)));
  }

  @Test
  void additionalLeafPredicateCanTreatAContainerAsALeaf() {
    List<String> nested = List.of("a", "b");

    TreeUtil.Flattened flattened =
        TreeUtil.flatten(List.of(nested, "tail"), value -> value == nested);

    assertEquals(List.of(nested, "tail"), flattened.leaves());
    assertEquals(List.of(nested, "replacement"),
        TreeUtil.unflatten(flattened.def(), List.of(nested, "replacement")));
  }

  @Test
  void registeredNodesParticipateInAllOperations() {
    TreeUtil.register(Box.class,
        value -> {
          Box box = (Box) value;
          return new TreeUtil.Node(List.of(box.value), box.name);
        },
        (name, children) -> new Box((String) name, children.getFirst()));

    Box box = new Box("weight", "w");
    assertEquals(List.of("w"), TreeUtil.leaves(box));
    assertEquals(new Box("weight", "w"), TreeUtil.unflatten(TreeUtil.structure(box), List.of("w")));
    assertEquals(new Box("weight", "w!"), TreeUtil.map((String value) -> value + "!", box));
    assertNotEquals(TreeUtil.structure(box), TreeUtil.structure(new Box("bias", "w")));
  }
}
