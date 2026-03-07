# Threshold Analysis Results

**Date**: 2026-03-07
**Task**: 2. Check ON/NEAR thresholds in spatial_relations.py

## Current Thresholds

```python
DEFAULT_THRESHOLDS = {
    "on_top_of": {"max_horizontal": 0.5, "min_vertical": 0.0, "max_vertical": 1.0},
    "above": {"max_horizontal": 1.0, "min_vertical": 0.1},
    "below": {"max_horizontal": 1.0, "max_vertical": -0.1},
    "next_to": {"max_distance": 1.5},
    "near": {"max_distance": 3.0},
    "inside": {"margin": 0.1},
    "between": {"max_distance_ratio": 0.3},
}
```

## Test Results Analysis

### ON Relation Tests
| Test | Query | Result | Status |
|------|-------|--------|--------|
| P1-C2-01 | "the throw_pillow on the sofa" | 1 | PASS |
| P1-C2-02 | "the vase on the coffee_table" | 1 | PASS |
| P2-C2-01 | "the cushion on the couch" | 0 | FAIL (parsing) |

### NEAR Relation Tests
| Test | Query | Result | Status |
|------|-------|--------|--------|
| P1-C2-03 | "the ottoman near the stool" | 1 | PASS |
| P1-C2-04 | "the armchair near the floor_lamp" | 1 | PASS |
| P1-C2-05 | "the decorative_bowl near the vase" | 1 | PASS |
| P2-C2-02 | "the pillow near the lamp" | 1 | PASS |
| P2-C2-03 | "the footstool near the rug" | 1 | PASS |

## Scene Object Analysis

The room0 scene contains 62 objects with categories:
- pillow: 10
- lamp: 8
- side table: 5
- hassock: 5 (mapped from "ottoman")
- stool: 4
- couch: 1 (mapped from "sofa")
- footrest: 3

The synonym mapping correctly handles:
- "sofa" → "couch"
- "ottoman" → "hassock", "footrest"
- "throw_pillow" → "pillow"

## P2-C2-01 Failure Analysis

**Query**: "the cushion on the couch"
**Expected**: >0 objects (pillow on couch)
**Actual**: 0 objects

**Root Cause**: LLM parser mapped "cushion" to `["sofa_seat_cushion"]` instead of `["throw_pillow", "pillow"]`.

This is a **parsing issue**, not a threshold issue. The `sofa_seat_cushion` category doesn't exist in the scene.

## Conclusion

**Thresholds are appropriate.** All spatial relation tests pass except P2-C2-01, which fails due to synonym mapping in the parser, not threshold settings.

### Recommendation
- No threshold changes needed
- Fix P2-C2-01 by updating parser synonym mapping (Task 2b)
