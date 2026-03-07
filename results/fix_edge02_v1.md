# EDGE-02 Fix Results

**Date**: 2026-03-07
**Task**: 1b. Fix EDGE-02: Remove `is_near` fallback from `is_inside` relation

## Change Made

Modified `conceptgraph/query_scene/spatial_relations.py` at line 475:

**Before:**
```python
if a_bbox is None:
    # Fall back to proximity check
    return self.is_near(target, anchor)
```

**After:**
```python
if a_bbox is None:
    # Cannot determine containment without bounding box
    # Do NOT fall back to proximity - "inside" requires bbox data
    return RelationResult(satisfies=False, score=0.0)
```

## Test Results

### EDGE-02: `"the sofa inside the ottoman"`
- **Before**: 1 object (FALSE POSITIVE - bbox fallback to `is_near`)
- **After**: 0 objects (CORRECT - impossible containment rejected)

### Overall Pass Rate
- **Before**: 57/59 (96.6%)
- **After**: 55/59 (93.2%)

### Analysis of Difference

The 2-test difference is NOT a regression from this fix:

1. **EDGE-02**: Now correctly returns 0 (this was the bug we fixed)
2. **P2-C2-01**: `"the cushion on the couch"` -> 0 objects
   - Root cause: LLM parser mapped "cushion" to `sofa_seat_cushion` instead of `throw_pillow`
   - No `sofa_seat_cushion` category exists in scene
   - This is LLM parsing variance, not related to `is_inside` fix

### Semantic Accuracy Assessment

| Test Case | Result | Expected | Status |
|-----------|--------|----------|--------|
| EDGE-01 | 0 | 0 (impossible) | TRUE NEGATIVE |
| EDGE-02 | 0 | 0 (impossible) | TRUE NEGATIVE (FIXED) |
| EDGE-03 | 0 | 0 (unlikely) | TRUE NEGATIVE |
| P2-C2-01 | 0 | >0 | FALSE NEGATIVE (parsing) |

**Semantic accuracy**: 58/59 = 98.3%
- 55 tests pass with >0 objects
- 3 tests correctly return 0 (EDGE-01/02/03)
- 1 test incorrectly returns 0 (P2-C2-01 - parsing issue)

## Conclusion

The EDGE-02 fix is successful. The `is_inside` relation no longer falls back to `is_near` when bbox data is unavailable, correctly rejecting impossible containment relations.

The P2-C2-01 failure is a separate issue related to LLM synonym mapping ("cushion" -> "sofa_seat_cushion" vs "throw_pillow").
