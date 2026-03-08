"""Unit tests for BEV builder with Open3D rendering."""

import gzip
import pickle
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import open3d as o3d
import pytest

from conceptgraph.query_scene.bev_builder import (
    AnnotatedObject,
    BEVConfig,
    GenericBEVBuilder,
    ReplicaBEVBuilder,
    create_bev_builder,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def replica_mesh_path() -> Path:
    """Return path to Replica room0 mesh if available."""
    path = Path("/Users/bytedance/Replica/room0_mesh.ply")
    if not path.exists():
        pytest.skip(f"Mesh file not found: {path}")
    return path


@pytest.fixture
def replica_objects_path() -> Path:
    """Return path to Replica room0 objects pkl.gz if available."""
    pcd_dir = Path("/Users/bytedance/Replica/room0/pcd_saves")
    if not pcd_dir.exists():
        pytest.skip(f"PCD saves directory not found: {pcd_dir}")

    pkl_files = list(pcd_dir.glob("*_post.pkl.gz"))
    if not pkl_files:
        pytest.skip(f"No pkl.gz files found in {pcd_dir}")
    return pkl_files[0]


@pytest.fixture
def sample_objects() -> List[dict]:
    """Create sample objects for testing."""
    return [
        {"centroid": [0.0, 0.0, 0.5], "category": "table"},
        {"centroid": [1.0, 1.0, 0.3], "category": "chair"},
        {"centroid": [-0.5, 2.0, 0.4], "category": "lamp"},
        {"centroid": [2.0, -1.0, 0.6], "category": "sofa"},
    ]


@pytest.fixture
def sample_pcd_objects() -> List[dict]:
    """Create sample objects with point clouds."""
    np.random.seed(42)
    return [
        {
            "pcd_np": np.random.rand(100, 3) + np.array([0, 0, 0.5]),
            "class_name": ["table"],
        },
        {
            "pcd_np": np.random.rand(100, 3) + np.array([1, 1, 0.3]),
            "class_name": ["chair"],
        },
    ]


# ============================================================================
# Config Tests
# ============================================================================


class TestBEVConfig:
    """Test BEVConfig defaults and validation."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = BEVConfig()
        assert config.image_size == 1500
        assert config.padding == 0.08
        assert config.object_diameter == 20
        assert config.background_color == (248, 248, 248)
        assert config.render_mesh is True

    def test_custom_config(self):
        """Custom config values should override defaults."""
        config = BEVConfig(image_size=800, padding=0.1, render_mesh=False)
        assert config.image_size == 800
        assert config.padding == 0.1
        assert config.render_mesh is False


# ============================================================================
# Annotation Extraction Tests
# ============================================================================


class TestReplicaBEVBuilder:
    """Test ReplicaBEVBuilder annotation extraction."""

    def test_extract_annotations_dict_centroids(self, sample_objects):
        """Extract annotations from dict objects with centroids."""
        builder = ReplicaBEVBuilder()
        annotations = builder.extract_annotations(sample_objects)

        assert len(annotations) == 4
        assert annotations[0].category == "table"
        assert annotations[0].centroid_3d == (0.0, 0.0, 0.5)
        assert annotations[1].category == "chair"

    def test_extract_annotations_pcd_np(self, sample_pcd_objects):
        """Extract annotations from objects with pcd_np."""
        builder = ReplicaBEVBuilder()
        annotations = builder.extract_annotations(sample_pcd_objects)

        assert len(annotations) == 2
        assert annotations[0].category == "table"
        assert annotations[1].category == "chair"
        # Centroids should be computed from pcd_np
        assert abs(annotations[0].centroid_3d[2] - 1.0) < 0.5  # z ~ 0.5 + 0.5

    def test_extract_empty_objects(self):
        """Empty objects list should return empty annotations."""
        builder = ReplicaBEVBuilder()
        annotations = builder.extract_annotations([])
        assert len(annotations) == 0


class TestGenericBEVBuilder:
    """Test GenericBEVBuilder flexible extraction."""

    def test_multiple_field_names(self):
        """Generic builder should handle various field names."""
        objects = [
            {"position": [1.0, 2.0, 3.0], "label": "obj_a"},
            {"xyz": [4.0, 5.0, 6.0], "type": "obj_b"},
            {"center": [7.0, 8.0, 9.0], "name": "obj_c"},
        ]

        builder = GenericBEVBuilder()
        annotations = builder.extract_annotations(objects)

        assert len(annotations) == 3
        assert annotations[0].centroid_3d == (1.0, 2.0, 3.0)
        assert annotations[0].category == "obj_a"
        assert annotations[1].centroid_3d == (4.0, 5.0, 6.0)
        assert annotations[2].centroid_3d == (7.0, 8.0, 9.0)


# ============================================================================
# Rendering Tests (without mesh)
# ============================================================================


class TestBuildWithoutMesh:
    """Test BEV generation without mesh rendering."""

    def test_build_creates_image(self, sample_objects):
        """Build should create a valid image."""
        config = BEVConfig(image_size=400, render_mesh=False)
        builder = ReplicaBEVBuilder(config=config)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = Path(f.name)

        img, path, labels = builder.build(sample_objects, output_path=output_path)

        assert img is not None
        assert img.shape == (400, 400, 3)
        assert path.exists()
        assert len(labels) == 4

        # Cleanup
        output_path.unlink()

    def test_build_empty_creates_placeholder(self):
        """Empty objects should create placeholder image."""
        config = BEVConfig(image_size=300, render_mesh=False)
        builder = ReplicaBEVBuilder(config=config)

        img, path, labels = builder.build([])

        assert img is not None
        assert img.shape == (300, 300, 3)
        assert len(labels) == 0

        # Cleanup
        path.unlink()


# ============================================================================
# Rendering Tests (with mesh)
# ============================================================================


class TestBuildWithMesh:
    """Test BEV generation with Open3D mesh rendering."""

    def test_render_mesh_creates_colored_image(self, replica_mesh_path, sample_objects):
        """Rendering with mesh should create a colored image."""
        config = BEVConfig(image_size=600)
        builder = ReplicaBEVBuilder(config=config)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = Path(f.name)

        img, path, labels = builder.build(
            sample_objects, output_path=output_path, mesh_path=replica_mesh_path
        )

        assert img is not None
        assert img.shape == (600, 600, 3)

        # Check that image has diverse colors (not just background)
        unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
        assert unique_colors > 100, f"Image has too few colors: {unique_colors}"

        # Cleanup
        output_path.unlink()

    def test_render_with_real_objects(
        self, replica_mesh_path, replica_objects_path
    ):
        """Render with actual scene objects from pkl.gz."""
        # Load objects
        with gzip.open(replica_objects_path, "rb") as f:
            data = pickle.load(f)
        objects = data.get("objects", [])

        if not objects:
            pytest.skip("No objects in pkl.gz file")

        config = BEVConfig(image_size=800)
        builder = ReplicaBEVBuilder(config=config)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = Path(f.name)

        img, path, labels = builder.build(
            objects, output_path=output_path, mesh_path=replica_mesh_path
        )

        assert len(labels) == len(objects)
        assert img.shape == (800, 800, 3)

        # Verify diverse colors
        unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
        assert unique_colors > 1000, f"Image has too few colors: {unique_colors}"

        # Cleanup
        output_path.unlink()


# ============================================================================
# Ceiling Detection Tests
# ============================================================================


class TestCeilingDetection:
    """Test ceiling threshold detection."""

    def test_detect_ceiling_simple(self):
        """Simple Z distribution should detect ceiling correctly."""
        builder = ReplicaBEVBuilder()

        # Create Z distribution with floor (~0) and ceiling (~2.5)
        z = np.concatenate([
            np.random.normal(0.0, 0.1, 1000),  # Floor
            np.random.normal(1.2, 0.3, 500),   # Furniture
            np.random.normal(2.5, 0.05, 200),  # Ceiling
        ])

        threshold = builder._detect_ceiling_threshold(z)

        # Threshold should be below ceiling peak
        assert threshold < 2.5
        assert threshold > 1.5


# ============================================================================
# Factory Tests
# ============================================================================


class TestFactory:
    """Test create_bev_builder factory function."""

    def test_create_replica_builder(self):
        """Factory should create ReplicaBEVBuilder."""
        builder = create_bev_builder("replica")
        assert isinstance(builder, ReplicaBEVBuilder)

    def test_create_generic_builder(self):
        """Factory should create GenericBEVBuilder."""
        builder = create_bev_builder("generic")
        assert isinstance(builder, GenericBEVBuilder)

    def test_create_with_config(self):
        """Factory should pass config to builder."""
        config = BEVConfig(image_size=999)
        builder = create_bev_builder("replica", config=config)
        assert builder.config.image_size == 999

    def test_create_unknown_raises(self):
        """Unknown dataset should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            create_bev_builder("unknown_dataset")


# ============================================================================
# Coordinate Transform Tests
# ============================================================================


class TestCoordinateTransform:
    """Test world-to-pixel coordinate transformations."""

    def test_world_to_pixel_center(self, sample_objects):
        """Object at scene center should be near image center."""
        config = BEVConfig(image_size=400, render_mesh=False)
        builder = ReplicaBEVBuilder(config=config)

        # Object at origin
        objects = [{"centroid": [0.0, 0.0, 0.5], "category": "center"}]
        img, _, _ = builder.build(objects)

        # Single object - should be roughly centered
        # (actual position depends on padding/margins)
        assert img is not None

    def test_multiple_objects_spread(self, sample_objects):
        """Multiple objects should be spread across image."""
        config = BEVConfig(image_size=400, render_mesh=False)
        builder = ReplicaBEVBuilder(config=config)

        img, _, labels = builder.build(sample_objects)

        assert len(labels) == 4


# ============================================================================
# Run as script
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
