"""
BEV Builder: Extensible Bird's Eye View rendering for different datasets.

This module provides a base class and dataset-specific implementations for
generating annotated BEV images suitable for multimodal LLM input.

Architecture:
    BaseBEVBuilder (abstract)
    ├── ReplicaBEVBuilder   - For Replica dataset scenes
    ├── ScanNetBEVBuilder   - For ScanNet dataset (future)
    └── CustomBEVBuilder    - For custom scenes with centroid data

Usage:
    # Using factory
    builder = create_bev_builder("replica", config=BEVConfig(image_size=1000))
    img, path, labels = builder.build(scene_objects)

    # Direct instantiation
    builder = ReplicaBEVBuilder()
    img, path, labels = builder.build(objects)
"""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class BEVConfig:
    """Configuration for BEV generation."""

    image_size: int = 1500  # Higher resolution for better quality
    padding: float = 0.08  # meters of padding around scene
    object_diameter: int = 20  # circle diameter in pixels
    font_scale: float = 0.45
    font_thickness: int = 1
    label_offset: Tuple[int, int] = (14, 5)
    background_color: Tuple[int, int, int] = (248, 248, 248)
    object_color: Tuple[int, int, int] = (50, 50, 210)  # Red-ish for visibility
    text_color: Tuple[int, int, int] = (25, 25, 25)
    border_color: Tuple[int, int, int] = (180, 180, 180)
    show_legend: bool = False  # Disable legend when mesh is shown
    show_title: bool = False  # Disable title when mesh is shown
    title: str = "Scene BEV (annotated)"
    max_labels_per_quadrant: int = 25  # Limit labels to avoid overcrowding
    category_colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    # Coordinate system options
    flip_y: bool = False  # Flip Y axis (for different coordinate conventions)
    swap_xy: bool = False  # Swap X and Y axes
    # Mesh rendering options
    mesh_path: Optional[Union[str, Path]] = None  # Path to mesh PLY file
    render_mesh: bool = True  # Whether to render mesh as background


@dataclass
class AnnotatedObject:
    """Object annotation for BEV visualization."""

    obj_id: int
    category: str
    centroid_3d: Tuple[float, float, float]  # (x, y, z) in world space
    centroid_2d: Tuple[float, float] = (0.0, 0.0)  # (x, y) projected for BEV
    pixel_pos: Tuple[int, int] = (0, 0)  # (x, y) in pixel space
    label: str = ""  # Display label (e.g., "001: sofa")
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra data


class BaseBEVBuilder(ABC):
    """
    Abstract base class for BEV image generation.

    Subclasses implement dataset-specific object extraction logic.
    The rendering pipeline is shared across all implementations.
    """

    def __init__(self, config: Optional[BEVConfig] = None):
        self.config = config or BEVConfig()

    @abstractmethod
    def extract_annotations(self, objects: List[Any]) -> List[AnnotatedObject]:
        """
        Extract annotations from dataset-specific object format.

        This method must be implemented by subclasses to handle
        the specific data format of each dataset.

        Args:
            objects: List of scene objects in dataset-specific format

        Returns:
            List of AnnotatedObject with centroid_3d and category
        """
        pass

    def build(
        self,
        objects: List[Any],
        output_path: Optional[Union[str, Path]] = None,
        mesh_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, Path, Dict[int, str]]:
        """
        Build annotated BEV image from scene objects.

        Args:
            objects: List of scene objects (format depends on dataset)
            output_path: Optional path to save image (creates temp if None)
            mesh_path: Optional path to mesh PLY file for background rendering

        Returns:
            Tuple of (image_array, image_path, obj_id_to_label_map)
        """
        if not objects:
            return self._create_empty_image(output_path)

        # Extract dataset-specific annotations
        annotations = self.extract_annotations(objects)
        if not annotations:
            return self._create_empty_image(output_path)

        # Project 3D centroids to 2D BEV
        self._project_to_2d(annotations)

        # Determine mesh path
        effective_mesh_path = mesh_path or self.config.mesh_path

        # Load mesh for bounds calculation if available
        mesh_vertices = None
        mesh_colors = None
        if effective_mesh_path and self.config.render_mesh:
            mesh_vertices, mesh_colors = self._load_mesh(effective_mesh_path)

        # Compute bounds and scale (from mesh if available, else from objects)
        if mesh_vertices is not None:
            transform = self._compute_transform_from_mesh(mesh_vertices)
        else:
            centroids = np.array([a.centroid_2d for a in annotations])
            transform = self._compute_transform(centroids)

        # Transform to pixel coordinates
        for ann in annotations:
            px, py = self._world_to_pixel(np.array(ann.centroid_2d), transform)
            ann.pixel_pos = (px, py)

        # Create and render image
        img = self._render_image(annotations, mesh_vertices, mesh_colors, transform)

        # Build label map
        obj_id_to_label = {ann.obj_id: ann.label for ann in annotations}

        # Save image
        output_path = self._save_image(img, output_path)

        return img, output_path, obj_id_to_label

    def _load_mesh(self, mesh_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load mesh vertices and colors from PLY file."""
        try:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            vertices = np.asarray(mesh.vertices)
            colors = np.asarray(mesh.vertex_colors)
            if len(vertices) > 0:
                return vertices, colors if len(colors) > 0 else None
        except Exception:
            pass
        return None, None

    def _compute_transform_from_mesh(self, vertices: np.ndarray) -> Dict[str, Any]:
        """Compute transform from mesh bounds."""
        padding = self.config.padding
        size = self.config.image_size

        # Use X, Y axes for BEV
        x = vertices[:, 0]
        y = vertices[:, 1]

        min_pt = np.array([x.min() - padding, y.min() - padding])
        max_pt = np.array([x.max() + padding, y.max() + padding])

        range_x = max_pt[0] - min_pt[0]
        range_y = max_pt[1] - min_pt[1]
        scale = (size * 0.92) / max(range_x, range_y)

        offset = np.array([
            (size - range_x * scale) / 2,
            (size - range_y * scale) / 2,
        ])

        return {
            "min_pt": min_pt,
            "max_pt": max_pt,
            "scale": scale,
            "offset": offset,
            "range_x": range_x,
            "range_y": range_y,
        }

    def _project_to_2d(self, annotations: List[AnnotatedObject]) -> None:
        """
        Project 3D centroids to 2D for BEV.

        Default: Use X, Y axes (top-down view from Z+).
        Override in subclass for different projections.
        """
        config = self.config
        for ann in annotations:
            x, y, z = ann.centroid_3d
            if config.swap_xy:
                x, y = y, x
            if config.flip_y:
                y = -y
            ann.centroid_2d = (x, y)

    def _compute_transform(
        self, centroids: np.ndarray
    ) -> Dict[str, Any]:
        """Compute world-to-pixel transformation parameters."""
        padding = self.config.padding
        size = self.config.image_size

        min_pt = centroids.min(axis=0) - padding
        max_pt = centroids.max(axis=0) + padding

        # Compute scale to fit in image
        range_x = max_pt[0] - min_pt[0]
        range_y = max_pt[1] - min_pt[1]
        scale = (size * 0.85) / max(range_x, range_y)

        # Compute offset to center
        offset = np.array([
            (size - range_x * scale) / 2,
            (size - range_y * scale) / 2,
        ])

        return {
            "min_pt": min_pt,
            "max_pt": max_pt,
            "scale": scale,
            "offset": offset,
        }

    def _world_to_pixel(
        self, point: np.ndarray, transform: Dict[str, Any]
    ) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        min_pt = transform["min_pt"]
        scale = transform["scale"]
        offset = transform["offset"]

        x = int((point[0] - min_pt[0]) * scale + offset[0])
        y = int((point[1] - min_pt[1]) * scale + offset[1])
        return x, y

    def _render_image(
        self,
        annotations: List[AnnotatedObject],
        mesh_vertices: Optional[np.ndarray] = None,
        mesh_colors: Optional[np.ndarray] = None,
        transform: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Render the BEV image with mesh background and annotations."""
        config = self.config
        size = config.image_size

        # Create blank image
        img = np.ones((size, size, 3), dtype=np.uint8)
        img[:] = config.background_color

        # Render mesh background if available
        if mesh_vertices is not None and transform is not None:
            self._render_mesh_background(img, mesh_vertices, mesh_colors, transform)

        # Draw objects
        self._draw_objects(img, annotations)

        # Draw labels with collision avoidance
        self._draw_labels(img, annotations)

        # Draw legend if enabled
        if config.show_legend:
            self._draw_legend(img, annotations)

        # Add title if enabled
        if config.show_title:
            cv2.putText(
                img,
                config.title,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                config.text_color,
                1,
            )

        return img

    def _detect_ceiling_threshold(self, z: np.ndarray) -> float:
        """Use histogram to detect ceiling Z threshold."""
        hist, bin_edges = np.histogram(z, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        z_range = z.max() - z.min()
        ceiling_start = z.max() - z_range * 0.25
        ceiling_mask = bin_centers > ceiling_start

        if not np.any(ceiling_mask):
            return z.max()

        ceiling_bins = hist[ceiling_mask]
        ceiling_centers = bin_centers[ceiling_mask]
        peak_idx = np.argmax(ceiling_bins)
        ceiling_peak_z = ceiling_centers[peak_idx]

        # Threshold: 0.15m below the ceiling peak
        return ceiling_peak_z - 0.15

    def _render_mesh_background(
        self,
        img: np.ndarray,
        vertices: np.ndarray,
        colors: Optional[np.ndarray],
        transform: Dict[str, Any],
    ) -> None:
        """Render mesh vertices as background with interpolation for smooth texture."""
        from scipy.interpolate import griddata

        size = self.config.image_size
        min_pt = transform["min_pt"]
        scale = transform["scale"]
        offset = transform["offset"]

        # Filter ceiling using histogram peak detection
        z = vertices[:, 2]
        ceiling_threshold = self._detect_ceiling_threshold(z)
        ceiling_mask = z < ceiling_threshold
        vertices = vertices[ceiling_mask]
        if colors is not None:
            colors = colors[ceiling_mask]

        x = vertices[:, 0]
        y = vertices[:, 1]

        # Transform to pixel coordinates
        px = ((x - min_pt[0]) * scale + offset[0]).astype(np.float32)
        py = ((y - min_pt[1]) * scale + offset[1]).astype(np.float32)
        py = size - 1 - py

        if colors is None:
            # Fallback: direct pixel assignment
            valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
            img[py[valid].astype(int), px[valid].astype(int)] = [128, 128, 128]
            return

        # Sample points for interpolation (limit for speed)
        max_points = 200000
        sample_step = max(1, len(px) // max_points)
        px_s = px[::sample_step]
        py_s = py[::sample_step]
        colors_s = colors[::sample_step]

        points = np.column_stack([px_s, py_s])
        grid_x, grid_y = np.mgrid[0:size, 0:size]

        # Interpolate each channel (RGB -> BGR)
        for c in range(3):
            channel = griddata(
                points,
                colors_s[:, 2 - c] * 255,
                (grid_x, grid_y),
                method='linear',
                fill_value=248,
            )
            img[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    def _draw_objects(self, img: np.ndarray, annotations: List[AnnotatedObject]) -> None:
        """Draw object circles on image with white outline for visibility."""
        config = self.config
        category_colors = config.category_colors or {}

        for ann in annotations:
            x, y = ann.pixel_pos

            # Get color (category-specific or default)
            color = category_colors.get(ann.category, config.object_color)

            # Draw white outline first for visibility on dark backgrounds
            cv2.circle(img, (x, y), config.object_diameter // 2 + 1, (255, 255, 255), 2)
            # Draw filled circle
            cv2.circle(img, (x, y), config.object_diameter // 2 - 1, color, -1)

    def _draw_labels(self, img: np.ndarray, annotations: List[AnnotatedObject]) -> None:
        """Draw labels with quadrant-based collision avoidance."""
        config = self.config
        size = config.image_size

        # Group by quadrant (0=top-right, 1=top-left, 2=bottom-left, 3=bottom-right)
        quadrants: Dict[int, List[AnnotatedObject]] = {0: [], 1: [], 2: [], 3: []}

        for ann in annotations:
            x, y = ann.pixel_pos
            q = 0
            if x < size // 2:
                q = 1 if y < size // 2 else 2
            else:
                q = 0 if y < size // 2 else 3
            quadrants[q].append(ann)

        # Draw labels per quadrant (limit to max_labels_per_quadrant)
        for q, anns in quadrants.items():
            # Sort by y position for vertical stacking
            anns.sort(key=lambda a: a.pixel_pos[1])
            anns = anns[:config.max_labels_per_quadrant]

            for ann in anns:
                x, y = ann.pixel_pos
                label = ann.label

                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, config.font_scale, config.font_thickness
                )

                # Offset direction based on quadrant
                ox, oy = config.label_offset
                if q in (1, 2):  # Left quadrants
                    lx = x - text_w - ox
                else:  # Right quadrants
                    lx = x + ox

                if q in (0, 1):  # Top quadrants
                    ly = y - oy
                else:  # Bottom quadrants
                    ly = y + text_h + oy

                # Clamp to image bounds
                lx = max(2, min(size - text_w - 2, lx))
                ly = max(text_h + 2, min(size - 2, ly))

                # Draw background rectangle
                cv2.rectangle(
                    img,
                    (lx - 1, ly - text_h - 1),
                    (lx + text_w + 1, ly + 2),
                    (255, 255, 255),
                    -1,
                )
                cv2.rectangle(
                    img,
                    (lx - 1, ly - text_h - 1),
                    (lx + text_w + 1, ly + 2),
                    (200, 200, 200),
                    1,
                )

                # Draw text
                cv2.putText(
                    img,
                    label,
                    (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    config.font_scale,
                    config.text_color,
                    config.font_thickness,
                )

    def _draw_legend(self, img: np.ndarray, annotations: List[AnnotatedObject]) -> None:
        """Draw category count legend."""
        config = self.config
        size = config.image_size

        # Count categories
        category_counts: Dict[str, int] = {}
        for ann in annotations:
            cat = ann.category
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Sort by count
        sorted_cats = sorted(category_counts.items(), key=lambda x: -x[1])[:10]

        # Draw legend in bottom-right
        legend_x = size - 150
        legend_y = size - 20 - len(sorted_cats) * 15

        cv2.putText(
            img,
            "Categories:",
            (legend_x, legend_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            config.text_color,
            1,
        )

        for i, (cat, count) in enumerate(sorted_cats):
            text = f"{cat[:12]}: {count}"
            cv2.putText(
                img,
                text,
                (legend_x, legend_y + 12 + i * 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (80, 80, 80),
                1,
            )

    def _create_empty_image(
        self, output_path: Optional[Union[str, Path]]
    ) -> Tuple[np.ndarray, Path, Dict[int, str]]:
        """Create empty placeholder image."""
        size = self.config.image_size
        img = np.ones((size, size, 3), dtype=np.uint8)
        img[:] = self.config.background_color

        cv2.putText(
            img,
            "No objects",
            (size // 2 - 50, size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.config.text_color,
            2,
        )

        output_path = self._save_image(img, output_path, prefix="bev_empty_")
        return img, output_path, {}

    def _save_image(
        self,
        img: np.ndarray,
        output_path: Optional[Union[str, Path]],
        prefix: str = "bev_",
    ) -> Path:
        """Save image to file."""
        if output_path is None:
            fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix=prefix)
            output_path = Path(tmp_path)
        else:
            output_path = Path(output_path)

        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return output_path


class ReplicaBEVBuilder(BaseBEVBuilder):
    """
    BEV builder for Replica dataset scenes.

    Expects objects in format:
    - Dict with 'pcd_np' (point cloud) and 'class_name' (list of categories)
    - Or SceneObject with .pcd_np, .centroid, .category attributes
    """

    def extract_annotations(self, objects: List[Any]) -> List[AnnotatedObject]:
        """Extract annotations from Replica scene objects."""
        annotations = []

        for i, obj in enumerate(objects):
            # Handle both dict and object-style access
            if isinstance(obj, dict):
                pcd_np = obj.get("pcd_np")
                centroid = obj.get("centroid")
                class_names = obj.get("class_name", [])
                category_attr = obj.get("category", "")
            else:
                pcd_np = getattr(obj, "pcd_np", None)
                centroid = getattr(obj, "centroid", None)
                class_names = getattr(obj, "class_name", [])
                category_attr = getattr(obj, "category", "")

            # Get centroid: prefer explicit centroid, then compute from pcd_np
            if centroid is not None:
                centroid_3d = tuple(float(c) for c in centroid[:3])
            elif pcd_np is not None and len(pcd_np) > 0:
                pcd_np = np.asarray(pcd_np)
                centroid_3d = tuple(float(c) for c in np.mean(pcd_np, axis=0)[:3])
            else:
                continue  # Skip objects without position data

            # Get category: prefer category attribute, then class_name list
            category = ""
            if category_attr:
                category = str(category_attr)
            elif class_names and len(class_names) > 0 and class_names[0]:
                category = str(class_names[0])
            else:
                category = f"obj_{i}"

            # Create label with global numbering
            label = f"{i:03d}: {category}"

            annotations.append(
                AnnotatedObject(
                    obj_id=i,
                    category=category,
                    centroid_3d=centroid_3d,
                    label=label,
                )
            )

        return annotations

    # Backward compatibility alias
    def generate(
        self,
        objects: List[Any],
        output_path: Optional[Union[str, Path]] = None,
        mesh_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, Path, Dict[int, str]]:
        """Alias for build() - for backward compatibility."""
        return self.build(objects, output_path, mesh_path=mesh_path)


class GenericBEVBuilder(BaseBEVBuilder):
    """
    Generic BEV builder for custom scenes.

    Flexible input format supporting both dicts and objects with:
    - centroid / centroid_3d / position / xyz
    - category / class_name / label / type
    """

    def extract_annotations(self, objects: List[Any]) -> List[AnnotatedObject]:
        """Extract annotations from generic scene objects."""
        annotations = []

        for i, obj in enumerate(objects):
            # Try to get centroid from various field names
            centroid_3d = self._get_centroid(obj, i)
            if centroid_3d is None:
                continue

            # Try to get category from various field names
            category = self._get_category(obj, i)

            # Create label
            label = f"{i:03d}: {category}"

            annotations.append(
                AnnotatedObject(
                    obj_id=i,
                    category=category,
                    centroid_3d=centroid_3d,
                    label=label,
                )
            )

        return annotations

    def _get_centroid(self, obj: Any, idx: int) -> Optional[Tuple[float, float, float]]:
        """Try to extract centroid from object using various field names."""
        field_names = ["centroid", "centroid_3d", "position", "xyz", "center", "loc", "location"]

        for name in field_names:
            val = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
            if val is not None:
                val = np.asarray(val).flatten()
                if len(val) >= 3:
                    return (float(val[0]), float(val[1]), float(val[2]))
                elif len(val) >= 2:
                    return (float(val[0]), float(val[1]), 0.0)

        # Try pcd_np as fallback
        pcd_np = obj.get("pcd_np") if isinstance(obj, dict) else getattr(obj, "pcd_np", None)
        if pcd_np is not None and len(pcd_np) > 0:
            pcd_np = np.asarray(pcd_np)
            c = np.mean(pcd_np, axis=0)
            if len(c) >= 3:
                return (float(c[0]), float(c[1]), float(c[2]))

        return None

    def _get_category(self, obj: Any, idx: int) -> str:
        """Try to extract category from object using various field names."""
        field_names = ["category", "class_name", "label", "type", "name", "tag"]

        for name in field_names:
            val = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
            if val is not None:
                if isinstance(val, list) and len(val) > 0:
                    return str(val[0])
                elif isinstance(val, str) and val:
                    return val

        return f"obj_{idx}"


# Alias for backward compatibility
SceneBEVGenerator = ReplicaBEVBuilder


def create_bev_builder(
    dataset: str = "replica",
    config: Optional[BEVConfig] = None,
) -> BaseBEVBuilder:
    """
    Factory function to create dataset-specific BEV builder.

    Args:
        dataset: Dataset name ("replica", "scannet", "generic")
        config: Optional BEVConfig

    Returns:
        BaseBEVBuilder instance

    Raises:
        ValueError: If dataset is not supported
    """
    builders = {
        "replica": ReplicaBEVBuilder,
        "generic": GenericBEVBuilder,
        # Future: "scannet": ScanNetBEVBuilder,
    }

    if dataset not in builders:
        supported = ", ".join(builders.keys())
        raise ValueError(f"Unknown dataset '{dataset}'. Supported: {supported}")

    return builders[dataset](config=config)
