from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from conceptgraph.query_scene.open_world_sample_builder import (
    TeacherQueryGenerator,
    bucket_counts,
    build_samples_for_scene,
    build_samples_from_assets,
    write_jsonl,
)
from conceptgraph.query_scene.query_structures import HypothesisOutputV1


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeClient:
    def __init__(self, scripted_results):
        self.scripted_results = list(scripted_results)
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        if not self.scripted_results:
            return _FakeResponse("find the object")
        current = self.scripted_results.pop(0)
        if isinstance(current, Exception):
            raise current
        return _FakeResponse(current)


class _FakeFactory:
    def __init__(self, scripts_by_model):
        self.scripts_by_model = scripts_by_model
        self.clients = {}

    def __call__(self, deployment_name=None, **kwargs):
        if deployment_name not in self.clients:
            scripted = self.scripts_by_model.get(deployment_name, ["find the object"])
            self.clients[deployment_name] = _FakeClient(scripted)
        return self.clients[deployment_name]


def _program(
    scene_id: str,
    idx: int,
    program_type: str,
    target: str,
    anchor: str | None = None,
    relation: str | None = None,
) -> dict:
    root = {
        "categories": [target],
        "attributes": [],
        "spatial_constraints": [],
        "select_constraint": None,
        "node_id": "root",
    }
    if relation and anchor:
        root["spatial_constraints"] = [
            {
                "relation": relation,
                "anchors": [
                    {
                        "categories": [anchor],
                        "attributes": [],
                        "spatial_constraints": [],
                        "select_constraint": None,
                        "node_id": "root_sc0_a0",
                    }
                ],
            }
        ]
    if program_type == "superlative" and anchor:
        root["select_constraint"] = {
            "constraint_type": "superlative",
            "metric": "distance",
            "order": "min",
            "reference": {
                "categories": [anchor],
                "attributes": [],
                "spatial_constraints": [],
                "select_constraint": None,
                "node_id": "root_sel_ref",
            },
            "position": None,
        }

    return {
        "scene_id": scene_id,
        "program_id": f"{scene_id}_prog_{idx:06d}",
        "program_type": program_type,
        "target_category": target,
        "anchor_categories": [anchor] if anchor else [],
        "relation": relation,
        "expect_unique": True,
        "program_hash": f"hash_{idx:04d}",
        "grounding_query": {
            "raw_query": "",
            "root": root,
            "expect_unique": True,
        },
    }


class TestOpenWorldSampleBuilder(unittest.TestCase):
    def test_bucket_counts(self) -> None:
        self.assertEqual(bucket_counts(10), {"direct": 4, "soft": 3, "hard": 3})
        self.assertEqual(bucket_counts(11), {"direct": 4, "soft": 3, "hard": 4})

    def test_teacher_query_generator_cache_and_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "teacher_cache.jsonl"
            factory = _FakeFactory(
                {
                    "gpt-5.2-2025-12-11": [RuntimeError("temporary"), "find the cushion near the couch"],
                    "gemini-3-pro-preview-new": ['{"query":"locate the pillow by the sofa"}'],
                }
            )

            generator = TeacherQueryGenerator(
                cache_path=cache_path,
                teacher_models=["gpt-5.2-2025-12-11", "gemini-3-pro-preview-new"],
                prompt_version="p_test_v1",
                temperature=0.2,
                seed=123,
                max_retries=2,
                llm_factory=factory,
            )

            prompt = "Write one realistic query"
            query1, meta1 = generator.generate_for_sample(
                scene_id="room0",
                program_hash="hash_x",
                prompt=prompt,
            )
            self.assertTrue(query1)
            self.assertFalse(meta1["all_failed"])
            self.assertEqual(len(meta1["candidates"]), 2)

            # First call should invoke models (gpt retries once).
            self.assertEqual(factory.clients["gpt-5.2-2025-12-11"].calls, 2)
            self.assertEqual(factory.clients["gemini-3-pro-preview-new"].calls, 1)

            # Second call hits cache; no additional invoke.
            query2, meta2 = generator.generate_for_sample(
                scene_id="room0",
                program_hash="hash_x",
                prompt=prompt,
            )
            self.assertEqual(query1, query2)
            self.assertTrue(all(c["cache_hit"] for c in meta2["candidates"]))
            self.assertEqual(factory.clients["gpt-5.2-2025-12-11"].calls, 2)
            self.assertEqual(factory.clients["gemini-3-pro-preview-new"].calls, 1)

            self.assertTrue(cache_path.exists())
            self.assertEqual(len(cache_path.read_text(encoding="utf-8").strip().splitlines()), 2)

    def test_build_samples_for_scene_distribution_and_hard_mask(self) -> None:
        scene_manifest = {
            "scene_id": "room0",
            "scene_categories": ["pillow", "throw_pillow", "sofa", "door", "side_table"],
        }
        programs = [
            _program("room0", 0, "simple", "pillow"),
            _program("room0", 1, "spatial", "pillow", anchor="sofa", relation="near"),
            _program("room0", 2, "superlative", "side_table", anchor="door", relation=None),
            _program("room0", 3, "spatial", "sofa", anchor="door", relation="near"),
            _program("room0", 4, "simple", "side_table"),
        ]

        parser_sft, counts = build_samples_for_scene(
            scene_manifest=scene_manifest,
            programs=programs,
            samples_per_scene=10,
            seed=7,
        )

        self.assertEqual(counts, {"direct": 4, "soft": 3, "hard": 3})
        self.assertEqual(len(parser_sft), 10)

        bucket_counter = {"direct": 0, "soft": 0, "hard": 0}
        for rec in parser_sft:
            bucket_counter[rec["bucket"]] += 1
            self.assertNotIn("gold_keyframes", rec)
            parsed = HypothesisOutputV1.model_validate(rec["target_output"])
            if rec["bucket"] == "hard":
                hidden = set(rec["mask_spec"]["hidden_categories"])
                parsed.validate_no_mask_leak(hidden)
                for hidden_cat in hidden:
                    self.assertNotIn(hidden_cat, rec["scene_categories"])
            else:
                self.assertNotIn("mask_spec", rec)

        self.assertEqual(bucket_counter, {"direct": 4, "soft": 3, "hard": 3})

    def test_build_samples_with_teacher_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "teacher_cache.jsonl"
            factory = _FakeFactory(
                {
                    "gpt-5.2-2025-12-11": ["find the cushion near the couch"] * 20,
                    "gemini-3-pro-preview-new": ["locate the pillow near the sofa"] * 20,
                }
            )
            teacher = TeacherQueryGenerator(
                cache_path=cache_path,
                teacher_models=["gpt-5.2-2025-12-11", "gemini-3-pro-preview-new"],
                prompt_version="p_test_v2",
                temperature=0.1,
                seed=7,
                max_retries=2,
                llm_factory=factory,
            )

            scene_manifest = {
                "scene_id": "room0",
                "scene_categories": ["pillow", "throw_pillow", "sofa", "door", "side_table"],
            }
            programs = [
                _program("room0", 0, "simple", "pillow"),
                _program("room0", 1, "spatial", "pillow", anchor="sofa", relation="near"),
                _program("room0", 2, "superlative", "side_table", anchor="door", relation=None),
                _program("room0", 3, "spatial", "sofa", anchor="door", relation="near"),
                _program("room0", 4, "simple", "side_table"),
            ]

            parser_sft, _ = build_samples_for_scene(
                scene_manifest=scene_manifest,
                programs=programs,
                samples_per_scene=6,
                seed=7,
                teacher_generator=teacher,
            )

            self.assertEqual(len(parser_sft), 6)
            for rec in parser_sft:
                self.assertIn("teacher_generation", rec)
                self.assertIn("selected_model", rec["teacher_generation"])
                self.assertIn("prompt_hash", rec["teacher_generation"])
                self.assertTrue(rec["user_query"])

    def test_build_samples_from_assets_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "scene_manifest.jsonl"
            pool_path = tmp_path / "query_program_pool.jsonl"
            out_dir = tmp_path / "out"

            scene_manifest = [
                {
                    "scene_id": "room0",
                    "scene_categories": ["pillow", "throw_pillow", "sofa", "door", "side_table"],
                }
            ]
            programs = [
                _program("room0", 0, "simple", "pillow"),
                _program("room0", 1, "spatial", "pillow", anchor="sofa", relation="near"),
                _program("room0", 2, "superlative", "side_table", anchor="door", relation=None),
                _program("room0", 3, "spatial", "sofa", anchor="door", relation="near"),
                _program("room0", 4, "simple", "side_table"),
            ]

            write_jsonl(scene_manifest, manifest_path)
            write_jsonl(programs, pool_path)

            summary = build_samples_from_assets(
                scene_manifest_path=manifest_path,
                query_program_pool_path=pool_path,
                output_dir=out_dir,
                samples_per_scene=10,
                seed=9,
            )

            self.assertEqual(summary["parser_sft_records"], 10)

            parser_file = out_dir / "parser_sft.jsonl"
            report_file = out_dir / "generation_report.md"

            self.assertTrue(parser_file.exists())
            self.assertFalse((out_dir / "retrieval_eval.jsonl").exists())
            self.assertTrue(report_file.exists())

            self.assertEqual(len(parser_file.read_text(encoding="utf-8").strip().splitlines()), 10)

            report_text = report_file.read_text(encoding="utf-8")
            self.assertIn("parser_sft_records: 10", report_text)
            self.assertNotIn("retrieval_eval_records", report_text)


if __name__ == "__main__":
    unittest.main()
