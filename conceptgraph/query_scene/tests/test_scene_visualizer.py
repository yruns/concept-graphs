"""
Tests for SceneBEVGenerator.

Tests verify that:
1. BEV images are generated correctly with object annotations
2. Labels follow the expected format (NNN: category)
3. Empty scenes are handled gracefully
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from conceptgraph.query_scene.scene_visualizer import (
    BEVConfig,
    SceneBEVGenerator,
)


class TestSceneBEVGenerator(unittest.TestCase):
    """Test SceneBEVGenerator functionality."""

    def test_bev_generation_basic(self):
        """Test basic BEV generation with mock objects."""
        # Create mock objects with point clouds
        objects = [
            {
                "pcd_np": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]),
                "class_name": ["sofa"],
            },
            {
                "pcd_np": np.array([[2.0, 2.0, 0.0], [2.5, 2.5, 0.5]]),
                "class_name": ["pillow"],
            },
            {
                "pcd_np": np.array([[-1.0, 3.0, 0.0], [-0.5, 3.5, 0.5]]),
                "class_name": ["door"],
            },
        ]

        generator = SceneBEVGenerator()
        img, path, obj_id_map = generator.generate(objects)

        # Verify output
        self.assertIsNotNone(img)
        self.assertEqual(img.ndim, 3)  # RGB image
        self.assertEqual(img.shape[2], 3)  # 3 channels
        self.assertTrue(path.exists())
        self.assertEqual(len(obj_id_map), 3)

        # Clean up
        path.unlink()

    def test_bev_label_format(self):
        """Test that labels follow 'NNN: category' format."""
        objects = [
            {
                "pcd_np": np.array([[0.0, 0.0, 0.0]]),
                "class_name": ["sofa"],
            },
            {
                "pcd_np": np.array([[1.0, 1.0, 0.0]]),
                "class_name": ["pillow"],
            },
        ]

        generator = SceneBEVGenerator()
        _, path, obj_id_map = generator.generate(objects)

        # Verify label format
        self.assertIn(0, obj_id_map)
        self.assertIn(1, obj_id_map)
        self.assertEqual(obj_id_map[0], "000: sofa")
        self.assertEqual(obj_id_map[1], "001: pillow")

        # Clean up
        path.unlink()

    def test_bev_empty_scene(self):
        """Test handling of empty object list."""
        generator = SceneBEVGenerator()
        img, path, obj_id_map = generator.generate([])

        # Should return placeholder image
        self.assertIsNotNone(img)
        self.assertEqual(img.ndim, 3)
        self.assertTrue(path.exists())
        self.assertEqual(len(obj_id_map), 0)

        # Clean up
        path.unlink()

    def test_bev_with_custom_config(self):
        """Test BEV generation with custom configuration."""
        config = BEVConfig(
            image_size=400,
            object_diameter=16,
            font_scale=0.5,
        )

        objects = [
            {
                "pcd_np": np.array([[0.0, 0.0, 0.0]]),
                "class_name": ["table"],
            },
        ]

        generator = SceneBEVGenerator(config=config)
        img, path, _ = generator.generate(objects)

        # Verify custom size
        self.assertEqual(img.shape[0], 400)
        self.assertEqual(img.shape[1], 400)

        # Clean up
        path.unlink()

    def test_bev_with_output_path(self):
        """Test saving to specific output path."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = Path(f.name)

        objects = [
            {
                "pcd_np": np.array([[0.0, 0.0, 0.0]]),
                "class_name": ["chair"],
            },
        ]

        generator = SceneBEVGenerator()
        _, returned_path, _ = generator.generate(objects, output_path=output_path)

        # Verify path matches
        self.assertEqual(returned_path, output_path)
        self.assertTrue(output_path.exists())

        # Clean up
        output_path.unlink()

    def test_bev_object_without_classname(self):
        """Test objects with missing class_name."""
        objects = [
            {
                "pcd_np": np.array([[0.0, 0.0, 0.0]]),
                # No class_name
            },
            {
                "pcd_np": np.array([[1.0, 1.0, 0.0]]),
                "class_name": [],  # Empty list
            },
        ]

        generator = SceneBEVGenerator()
        _, path, obj_id_map = generator.generate(objects)

        # Should use fallback category names
        self.assertEqual(len(obj_id_map), 2)
        self.assertTrue("obj_0" in obj_id_map[0] or "000:" in obj_id_map[0])

        # Clean up
        path.unlink()

    def test_bev_skips_objects_without_pcd(self):
        """Test that objects without point cloud are skipped."""
        objects = [
            {
                "pcd_np": np.array([[0.0, 0.0, 0.0]]),
                "class_name": ["sofa"],
            },
            {
                "pcd_np": None,  # No point cloud
                "class_name": ["ghost_object"],
            },
            {
                "pcd_np": np.array([]),  # Empty array
                "class_name": ["empty_object"],
            },
        ]

        generator = SceneBEVGenerator()
        _, path, obj_id_map = generator.generate(objects)

        # Only the first object should be included
        self.assertEqual(len(obj_id_map), 1)
        self.assertIn(0, obj_id_map)

        # Clean up
        path.unlink()


if __name__ == "__main__":
    unittest.main()
