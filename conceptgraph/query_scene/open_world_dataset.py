"""Utilities for building open-world query training assets."""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_RELATIONS = ["near", "on", "next_to", "above", "below"]


def detect_pcd_file(scene_path: Path) -> Optional[Path]:
    """Auto-detect the best PCD file for a scene."""
    pcd_dir = scene_path / "pcd_saves"
    candidates = list(pcd_dir.glob("*ram*_post.pkl.gz"))
    if not candidates:
        candidates = list(pcd_dir.glob("*_post.pkl.gz"))
    if not candidates:
        candidates = list(pcd_dir.glob("*.pkl.gz"))
    return candidates[0] if candidates else None


def detect_affordance_file(scene_path: Path) -> Optional[Path]:
    """Auto-detect object affordance file for a scene."""
    detect_path = scene_path / "sg_cache_detect" / "object_affordances.json"
    if detect_path.exists():
        return detect_path
    fallback = scene_path / "sg_cache" / "object_affordances.json"
    return fallback if fallback.exists() else None


def _infer_category(class_names: Sequence[Any], obj_id: int) -> str:
    """Infer a stable category from class_name list."""
    cleaned = []
    for name in class_names:
        text = str(name).strip()
        if not text:
            continue
        if text.lower() in {"item", "object", "none"}:
            continue
        cleaned.append(text)

    if cleaned:
        return Counter(cleaned).most_common(1)[0][0]
    if class_names:
        return str(class_names[0]).strip() or f"object_{obj_id}"
    return f"object_{obj_id}"


def load_scene_categories(scene_path: Path) -> Tuple[List[str], int, Path, Optional[Path]]:
    """
    Load scene categories with optional affordance override.

    Returns:
        categories, num_objects, pcd_file, affordance_file
    """
    pcd_file = detect_pcd_file(scene_path)
    if pcd_file is None:
        raise FileNotFoundError(f"No pcd file found under {scene_path / 'pcd_saves'}")

    with gzip.open(pcd_file, "rb") as f:
        payload = pickle.load(f)

    raw_objects = payload.get("objects", [])
    categories_by_id: Dict[int, str] = {}
    for obj_id, obj in enumerate(raw_objects):
        class_names = obj.get("class_name", [])
        categories_by_id[obj_id] = _infer_category(class_names, obj_id)

    affordance_file = detect_affordance_file(scene_path)
    if affordance_file is not None:
        with open(affordance_file, "r", encoding="utf-8") as f:
            affordances = json.load(f)
        for item in affordances:
            obj_id = item.get("id")
            tag = str(item.get("object_tag", "")).strip()
            if obj_id in categories_by_id and tag:
                categories_by_id[obj_id] = tag

    categories = sorted(set(categories_by_id.values()))
    return categories, len(raw_objects), pcd_file, affordance_file


def count_frames(scene_path: Path) -> int:
    """Count RGB frames under scene/results."""
    return len(list((scene_path / "results").glob("frame*.jpg")))


def build_scene_manifest_entry(
    scene_path: Path,
    scene_id: Optional[str] = None,
    stride: int = 5,
) -> Dict[str, Any]:
    """Build one scene manifest record."""
    categories, num_objects, pcd_file, affordance_file = load_scene_categories(scene_path)
    visibility_index = scene_path / "indices" / "visibility_index.pkl"

    return {
        "scene_id": scene_id or scene_path.name,
        "scene_path": str(scene_path.resolve()),
        "stride": stride,
        "pcd_file": str(pcd_file.resolve()),
        "affordance_file": str(affordance_file.resolve()) if affordance_file else None,
        "visibility_index_file": str(visibility_index.resolve()) if visibility_index.exists() else None,
        "num_objects": num_objects,
        "num_frames": count_frames(scene_path),
        "num_categories": len(categories),
        "scene_categories": categories,
    }


def _make_grounding_query(
    target: str,
    relation: Optional[str] = None,
    anchor: Optional[str] = None,
    expect_unique: bool = True,
    use_nearest: bool = False,
) -> Dict[str, Any]:
    root: Dict[str, Any] = {
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

    if use_nearest and anchor:
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
        "raw_query": "",
        "root": root,
        "expect_unique": expect_unique,
    }


def _program_hash(program_signature: Dict[str, Any]) -> str:
    """Stable hash for de-dup and split control."""
    canonical = json.dumps(program_signature, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def generate_query_program_pool(
    scene_manifest: Dict[str, Any],
    max_programs_per_scene: int = 300,
    relations: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate deterministic query programs for one scene.

    Program types:
    - simple: target only
    - spatial: target + relation + anchor
    - superlative: nearest target to anchor
    """
    categories = sorted(scene_manifest.get("scene_categories", []))
    scene_id = scene_manifest["scene_id"]
    relations = list(relations or DEFAULT_RELATIONS)

    records: List[Dict[str, Any]] = []
    seen_hashes = set()

    def add_record(program_type: str, target: str, anchor: Optional[str], relation: Optional[str], expect_unique: bool, use_nearest: bool = False) -> None:
        signature = {
            "program_type": program_type,
            "target": target,
            "anchor": anchor,
            "relation": relation,
            "expect_unique": expect_unique,
            "use_nearest": use_nearest,
        }
        program_hash = _program_hash(signature)
        if program_hash in seen_hashes:
            return

        grounding_query = _make_grounding_query(
            target=target,
            relation=relation,
            anchor=anchor,
            expect_unique=expect_unique,
            use_nearest=use_nearest,
        )
        record = {
            "scene_id": scene_id,
            "program_type": program_type,
            "target_category": target,
            "anchor_categories": [anchor] if anchor else [],
            "relation": relation,
            "expect_unique": expect_unique,
            "program_hash": program_hash,
            "grounding_query": grounding_query,
        }
        seen_hashes.add(program_hash)
        records.append(record)

    for cat in categories:
        add_record("simple", target=cat, anchor=None, relation=None, expect_unique=True)
        if len(records) >= max_programs_per_scene:
            break

    for target in categories:
        for anchor in categories:
            if target == anchor:
                continue
            for relation in relations:
                add_record(
                    "spatial",
                    target=target,
                    anchor=anchor,
                    relation=relation,
                    expect_unique=True,
                )
                if len(records) >= max_programs_per_scene:
                    return records

    for target in categories:
        for anchor in categories:
            if target == anchor:
                continue
            add_record(
                "superlative",
                target=target,
                anchor=anchor,
                relation=None,
                expect_unique=True,
                use_nearest=True,
            )
            if len(records) >= max_programs_per_scene:
                return records

    return records


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: Path) -> int:
    """Write records to JSONL and return count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
