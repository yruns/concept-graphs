from __future__ import annotations

import gzip
import pickle
import tempfile
from pathlib import Path
import unittest

from conceptgraph.query_scene.open_world_dataset import (
    build_scene_manifest_entry,
    generate_query_program_pool,
    write_jsonl,
)
from conceptgraph.scripts.build_open_world_dataset_assets import _parse_scene_arg


class TestOpenWorldDataset(unittest.TestCase):
    def _build_mock_scene(self, root: Path) -> Path:
        scene = root / "room_mock"
        (scene / "pcd_saves").mkdir(parents=True)
        (scene / "sg_cache_detect").mkdir(parents=True)
        (scene / "results").mkdir(parents=True)

        payload = {
            "objects": [
                {"class_name": ["sofa", "sofa"]},
                {"class_name": ["pillow"]},
                {"class_name": []},
            ]
        }
        pcd_file = scene / "pcd_saves" / "full_pcd_ram_mock_post.pkl.gz"
        with gzip.open(pcd_file, "wb") as f:
            pickle.dump(payload, f)

        affordance_file = scene / "sg_cache_detect" / "object_affordances.json"
        affordance_file.write_text(
            """[
  {"id": 1, "object_tag": "throw_pillow"}
]
""",
            encoding="utf-8",
        )

        (scene / "results" / "frame000000.jpg").write_text("x", encoding="utf-8")
        (scene / "results" / "frame000005.jpg").write_text("x", encoding="utf-8")
        return scene

    def test_build_scene_manifest_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene = self._build_mock_scene(Path(tmp))
            manifest = build_scene_manifest_entry(scene, scene_id="room_mock", stride=5)

            self.assertEqual(manifest["scene_id"], "room_mock")
            self.assertEqual(manifest["num_objects"], 3)
            self.assertEqual(manifest["num_frames"], 2)
            self.assertEqual(manifest["stride"], 5)
            self.assertEqual(
                manifest["scene_categories"],
                ["object_2", "sofa", "throw_pillow"],
            )

    def test_generate_query_program_pool_is_deterministic(self) -> None:
        manifest = {
            "scene_id": "room0",
            "scene_categories": ["door", "sofa", "throw_pillow"],
        }

        pool_a = generate_query_program_pool(manifest, max_programs_per_scene=40)
        pool_b = generate_query_program_pool(manifest, max_programs_per_scene=40)

        self.assertEqual(pool_a, pool_b)
        self.assertTrue(pool_a)

        hashes = [item["program_hash"] for item in pool_a]
        self.assertEqual(len(hashes), len(set(hashes)))

        for item in pool_a:
            self.assertIn("program_type", item)
            self.assertIn("target_category", item)
            self.assertIn("grounding_query", item)
            self.assertIn("root", item["grounding_query"])

    def test_write_jsonl_returns_record_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            count = write_jsonl([{"a": 1}, {"a": 2}], path)
            self.assertEqual(count, 2)
            self.assertEqual(len(path.read_text(encoding="utf-8").strip().splitlines()), 2)

    def test_parse_scene_arg(self) -> None:
        scene_id, scene_path = _parse_scene_arg("room0=/tmp/room0")
        self.assertEqual(scene_id, "room0")
        self.assertEqual(scene_path, Path("/tmp/room0").resolve())


if __name__ == "__main__":
    unittest.main()
