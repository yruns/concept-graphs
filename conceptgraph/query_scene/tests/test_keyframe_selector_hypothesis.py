from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from conceptgraph.query_scene.keyframe_selector import KeyframeSelector
from conceptgraph.query_scene.query_executor import ExecutionResult
from conceptgraph.query_scene.query_structures import HypothesisKind


def _minimal_selector(tmp_path: Path) -> KeyframeSelector:
    selector = KeyframeSelector.__new__(KeyframeSelector)
    selector.scene_categories = ["pillow", "sofa", "door", "side_table"]
    selector.stride = 5
    selector.scene_path = tmp_path
    selector.image_paths = []
    selector.objects = []
    selector.object_features = None
    selector._query_executor = None
    selector._query_parser = None
    selector._relation_checker = None
    selector.llm_model = "test"
    return selector


def _grounding_query_dict(category: str) -> dict:
    return {
        "raw_query": f"find {category}",
        "root": {
            "categories": [category],
            "attributes": [],
            "spatial_constraints": [],
            "select_constraint": None,
            "node_id": "root",
        },
        "expect_unique": True,
    }


class TestKeyframeSelectorHypothesis(unittest.TestCase):
    def test_normalize_hypothesis_output_supports_legacy_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            selector = _minimal_selector(Path(tmp))
            legacy_hypothesis = {
                "grounding_query": _grounding_query_dict("pillow"),
                "lexical_hints": ["cushion"],
            }
            normalized = selector.normalize_hypothesis_output(legacy_hypothesis)
            self.assertEqual(normalized.parse_mode.value, "single")
            self.assertEqual(normalized.hypotheses[0].kind.value, "direct")
            self.assertEqual(normalized.hypotheses[0].grounding_query.root.category, "pillow")

            legacy_grounding_query = _grounding_query_dict("sofa")
            normalized2 = selector.normalize_hypothesis_output(legacy_grounding_query)
            self.assertEqual(normalized2.parse_mode.value, "single")
            self.assertEqual(normalized2.hypotheses[0].grounding_query.root.category, "sofa")

    def test_normalize_hypothesis_output_sorts_by_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            selector = _minimal_selector(Path(tmp))
            payload = {
                "format_version": "hypothesis_output_v1",
                "parse_mode": "multi",
                "hypotheses": [
                    {
                        "kind": "proxy",
                        "rank": 2,
                        "grounding_query": _grounding_query_dict("sofa"),
                        "lexical_hints": ["proxy"],
                    },
                    {
                        "kind": "direct",
                        "rank": 1,
                        "grounding_query": _grounding_query_dict("UNKNOW"),
                        "lexical_hints": [],
                    },
                ],
            }

            normalized = selector.normalize_hypothesis_output(payload)
            self.assertEqual([h.rank for h in normalized.hypotheses], [1, 2])

    def test_to_grounding_query_rejects_invalid_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            selector = _minimal_selector(Path(tmp))
            with self.assertRaises(ValueError):
                selector.to_grounding_query({"grounding_query": "bad"})

    def test_execute_hypotheses_proxy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            selector = _minimal_selector(Path(tmp))
            payload = {
                "format_version": "hypothesis_output_v1",
                "parse_mode": "multi",
                "hypotheses": [
                    {
                        "kind": "direct",
                        "rank": 1,
                        "grounding_query": _grounding_query_dict("UNKNOW"),
                        "lexical_hints": [],
                    },
                    {
                        "kind": "proxy",
                        "rank": 2,
                        "grounding_query": _grounding_query_dict("sofa"),
                        "lexical_hints": ["couch"],
                    },
                ],
            }

            def fake_execute_query(grounding_query):
                if grounding_query.root.category == "sofa":
                    return ExecutionResult(node_id="ok", matched_objects=[object()])
                return ExecutionResult(node_id="none", matched_objects=[])

            selector.execute_query = fake_execute_query  # type: ignore[method-assign]

            status, hypothesis, result = selector.execute_hypotheses(payload)
            self.assertEqual(status, "proxy_grounded")
            self.assertIsNotNone(hypothesis)
            self.assertEqual(hypothesis.kind, HypothesisKind.PROXY)
            self.assertFalse(result.is_empty)

    def test_execute_hypotheses_checks_hidden_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            selector = _minimal_selector(Path(tmp))
            payload = {
                "format_version": "hypothesis_output_v1",
                "parse_mode": "single",
                "hypotheses": [
                    {
                        "kind": "direct",
                        "rank": 1,
                        "grounding_query": _grounding_query_dict("pillow"),
                        "lexical_hints": [],
                    }
                ],
            }

            selector.execute_query = lambda gq: ExecutionResult(node_id="ok", matched_objects=[object()])  # type: ignore[method-assign]

            with self.assertRaises(ValueError):
                selector.execute_hypotheses(payload, hidden_categories={"pillow"})

    def test_map_and_resolve_keyframe_with_adjacent_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            selector = _minimal_selector(tmp_path)

            results_dir = tmp_path / "results"
            results_dir.mkdir(parents=True)

            # Request view=3 -> frame 15 is missing; fallback should find view=2 -> frame 10.
            fallback_frame = results_dir / "frame000010.jpg"
            fallback_frame.write_text("x", encoding="utf-8")

            path, resolved_view = selector._resolve_keyframe_path(3)
            self.assertEqual(path, fallback_frame)
            self.assertEqual(resolved_view, 2)
            self.assertEqual(selector.map_view_to_frame(3), 15)

    def test_resolve_keyframe_uses_image_path_when_results_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            selector = _minimal_selector(tmp_path)

            img = tmp_path / "sampled_frame.jpg"
            img.write_text("x", encoding="utf-8")
            selector.image_paths = [img]

            path, resolved_view = selector._resolve_keyframe_path(0)
            self.assertEqual(path, img)
            self.assertEqual(resolved_view, 0)


if __name__ == "__main__":
    unittest.main()
