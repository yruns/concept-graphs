#!/usr/bin/env python3
"""End-to-end evaluation pipeline for QueryParser and KeyframeSelector.

Stages:
1. Generate EvaluationCases with ground truth (QueryCaseGenerator)
2. Run QueryParser + KeyframeSelector inference
3. Evaluate with LLMEvaluatorV2
4. Generate summary report

Usage:
    # Pilot on room0 (15 cases)
    python -m conceptgraph.query_scene.examples.run_e2e_evaluation --mode pilot

    # Smoke test (2 scenes, 30 cases)
    python -m conceptgraph.query_scene.examples.run_e2e_evaluation --mode smoke

    # Full evaluation (8 scenes, 100 cases)
    python -m conceptgraph.query_scene.examples.run_e2e_evaluation --mode full
"""

import argparse
import gzip
import json
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:7} | {message}")


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class EvalConfig:
    """Frozen configuration for reproducibility."""

    # Run metadata
    run_id: str = ""
    mode: str = "pilot"  # pilot, smoke, full
    random_seed: int = 42
    timestamp: str = ""

    # Model versions
    generator_model: str = "gemini-2.5-pro"
    parser_model: str = "gemini-2.5-pro"
    evaluator_model: str = "gemini-2.5-pro"
    prompt_version: str = "v2"

    # Scene allocation
    scenes: List[str] = field(default_factory=list)
    cases_per_scene: Dict[str, int] = field(default_factory=dict)
    total_cases: int = 0

    # Difficulty distribution
    difficulty_distribution: Dict[str, float] = field(
        default_factory=lambda: {"easy": 0.4, "medium": 0.4, "hard": 0.2}
    )
    target_distribution: Dict[int, float] = field(
        default_factory=lambda: {1: 0.7, 2: 0.25, 3: 0.05}
    )

    # Execution
    max_workers: int = 3
    max_retries: int = 4  # Codex: increased for 100-case robustness
    retry_backoff_base: float = 2.0
    checkpoint_interval: int = 5  # Codex: more frequent checkpoints

    def __post_init__(self):
        if not self.run_id:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def apply_seed(self):
        """Apply random seed for reproducibility."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        # Try to set torch seed if available
        try:
            import torch
            torch.manual_seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)
        except ImportError:
            pass


def create_config(mode: str, replica_root: Path) -> EvalConfig:
    """Create configuration based on execution mode."""
    all_scenes = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]

    # Filter to available scenes
    available_scenes = [s for s in all_scenes if (replica_root / s / "indices" / "visibility_index.pkl").exists()]

    if mode == "pilot":
        scenes = ["room0"]
        total_cases = 15
    elif mode == "smoke":
        scenes = ["room0", "office0"]
        total_cases = 30
    else:  # full
        scenes = available_scenes
        total_cases = 100
        # Codex-recommended allocation: oversample hard scenes (room0/office0)
        codex_allocation = {
            "room0": 14, "office0": 14,
            "room1": 12, "room2": 12,
            "office1": 12, "office2": 12, "office3": 12, "office4": 12
        }

    # Allocate cases per scene
    if mode == "full" and len(scenes) == 8:
        # Use Codex-recommended allocation for full 8-scene runs
        cases_per_scene = {s: codex_allocation.get(s, 12) for s in scenes}
    else:
        # Default: floor + distribute remainder
        floor = total_cases // len(scenes)
        remainder = total_cases % len(scenes)
        cases_per_scene = {s: floor + (1 if i < remainder else 0) for i, s in enumerate(scenes)}

    return EvalConfig(
        mode=mode,
        scenes=scenes,
        cases_per_scene=cases_per_scene,
        total_cases=total_cases,
    )


# ============================================================================
# Pipeline Stages
# ============================================================================


@dataclass
class StageResult:
    """Result of a pipeline stage."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    retries: int = 0


@dataclass
class CaseResult:
    """Complete result for a single evaluation case."""

    case_id: str
    scene: str
    query: str

    # Stage results
    generation: Optional[StageResult] = None
    parsing: Optional[StageResult] = None
    selection: Optional[StageResult] = None
    evaluation: Optional[StageResult] = None

    # Final scores (if successful)
    parse_score: Optional[float] = None
    selector_score: Optional[float] = None
    overall_score: Optional[float] = None

    # Status
    status: str = "pending"  # pending, success, failed
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None


class E2EPipeline:
    """End-to-end evaluation pipeline."""

    def __init__(self, config: EvalConfig, replica_root: Path, output_dir: Path):
        self.config = config
        self.replica_root = replica_root
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Apply random seed
        config.apply_seed()

        # Results storage
        self.results: List[CaseResult] = []
        self.stage_stats = {
            "generation": {"attempted": 0, "success": 0, "failed": 0},
            "parsing": {"attempted": 0, "success": 0, "failed": 0},
            "selection": {"attempted": 0, "success": 0, "failed": 0},
            "evaluation": {"attempted": 0, "success": 0, "failed": 0},
        }

        # Cached selectors per scene (reuse to avoid repeated loading)
        self._selectors: Dict[str, Any] = {}
        # Cached evaluator (reuse across all cases)
        self._evaluator = None

    def run(self) -> Dict[str, Any]:
        """Run the complete pipeline."""
        logger.info(f"Starting E2E evaluation: mode={self.config.mode}, total_cases={self.config.total_cases}")
        logger.info(f"Scenes: {self.config.scenes}")
        logger.info(f"Output dir: {self.output_dir}")

        # Save config
        self._save_config()

        # Stage 1: Generate cases for each scene
        all_cases = []
        for scene in self.config.scenes:
            num_cases = self.config.cases_per_scene[scene]
            logger.info(f"Generating {num_cases} cases for {scene}...")
            cases = self._generate_cases(scene, num_cases)
            all_cases.extend(cases)
            self._checkpoint("generation", scene, cases)

        logger.info(f"Generated {len(all_cases)} cases total")

        # Stage 2 & 3: Parse + Select for each case
        for i, case_result in enumerate(all_cases):
            if case_result.status == "failed":
                continue

            logger.info(f"[{i+1}/{len(all_cases)}] Processing: {case_result.query[:50]}...")

            # Get or create selector for this scene (reuse per scene)
            selector = self._get_selector(case_result.scene)
            if selector is None:
                case_result.status = "failed"
                case_result.failure_stage = "parsing"
                case_result.failure_reason = "Failed to create KeyframeSelector"
                self.stage_stats["parsing"]["attempted"] += 1
                self.stage_stats["parsing"]["failed"] += 1
                continue

            # Parse
            parse_result = self._run_parser(case_result, selector)
            case_result.parsing = parse_result
            self.stage_stats["parsing"]["attempted"] += 1

            if not parse_result.success:
                case_result.status = "failed"
                case_result.failure_stage = "parsing"
                case_result.failure_reason = parse_result.error
                self.stage_stats["parsing"]["failed"] += 1
                continue
            self.stage_stats["parsing"]["success"] += 1

            # Select
            select_result = self._run_selector(case_result, selector, parse_result.data)
            case_result.selection = select_result
            self.stage_stats["selection"]["attempted"] += 1

            if not select_result.success:
                case_result.status = "failed"
                case_result.failure_stage = "selection"
                case_result.failure_reason = select_result.error
                self.stage_stats["selection"]["failed"] += 1
                continue
            self.stage_stats["selection"]["success"] += 1

            # Checkpoint every N cases
            if (i + 1) % self.config.checkpoint_interval == 0:
                self._checkpoint("inference", f"checkpoint_{i+1}", all_cases)

        # Stage 4: Evaluate all successful cases
        successful_cases = [c for c in all_cases if c.status != "failed"]
        logger.info(f"Evaluating {len(successful_cases)} successful cases...")

        for i, case_result in enumerate(successful_cases):
            logger.info(f"[{i+1}/{len(successful_cases)}] Evaluating: {case_result.query[:50]}...")

            eval_result = self._run_evaluator(case_result)
            case_result.evaluation = eval_result
            self.stage_stats["evaluation"]["attempted"] += 1

            if eval_result.success:
                case_result.status = "success"
                # Extract scores
                eval_data = eval_result.data
                case_result.parse_score = eval_data.get("parse_score")
                case_result.selector_score = eval_data.get("selector_score")
                case_result.overall_score = eval_data.get("overall_score")
                self.stage_stats["evaluation"]["success"] += 1
            else:
                case_result.status = "failed"
                case_result.failure_stage = "evaluation"
                case_result.failure_reason = eval_result.error
                self.stage_stats["evaluation"]["failed"] += 1

        self.results = all_cases

        # Generate report
        report = self._generate_report()
        self._save_results(report)

        return report

    def _get_selector(self, scene: str):
        """Get or create KeyframeSelector for a scene (cached per scene)."""
        if scene not in self._selectors:
            from conceptgraph.query_scene.keyframe_selector import KeyframeSelector

            scene_path = self.replica_root / scene
            try:
                # Must pass llm_model for query parsing
                selector = KeyframeSelector.from_scene_path(
                    scene_path,
                    llm_model=self.config.parser_model,
                    use_pool=True,
                )
                self._selectors[scene] = selector
            except Exception as e:
                logger.error(f"Failed to create selector for {scene}: {e}")
                return None

        return self._selectors[scene]

    def _get_evaluator(self):
        """Get or create LLMEvaluatorV2 (cached, reused across cases)."""
        if self._evaluator is None:
            from conceptgraph.query_scene.llm_evaluator_v2 import (
                BatchEvaluatorConfig,
                LLMEvaluatorV2,
            )

            eval_config = BatchEvaluatorConfig(
                max_workers=self.config.max_workers,
                max_retries=self.config.max_retries,
                retry_backoff_base=self.config.retry_backoff_base,
                model_name=self.config.evaluator_model,
            )
            self._evaluator = LLMEvaluatorV2(config=eval_config)

        return self._evaluator

    def _generate_cases(self, scene: str, num_cases: int) -> List[CaseResult]:
        """Generate evaluation cases for a scene."""
        from conceptgraph.query_scene.query_case_generator import QueryCaseGenerator

        scene_path = self.replica_root / scene
        results = []

        try:
            generator = QueryCaseGenerator(
                scene_path=scene_path,
                temperature=0.7,
                max_workers=self.config.max_workers,
            )

            batch = generator.generate_cases(
                num_cases=num_cases,
                target_distribution=self.config.target_distribution,
            )

            for i, case in enumerate(batch.cases):
                case_result = CaseResult(
                    case_id=f"{scene}_{i:03d}",
                    scene=scene,
                    query=case.query,
                    generation=StageResult(
                        success=True,
                        data=case.model_dump(),
                        duration_seconds=0,
                    ),
                )
                results.append(case_result)
                self.stage_stats["generation"]["attempted"] += 1
                self.stage_stats["generation"]["success"] += 1

            # Check if fewer cases than requested (partial failure)
            if len(batch.cases) < num_cases:
                missing = num_cases - len(batch.cases)
                logger.warning(f"Generator returned {len(batch.cases)}/{num_cases} cases for {scene}")
                self.stage_stats["generation"]["attempted"] += missing
                self.stage_stats["generation"]["failed"] += missing

        except Exception as e:
            logger.error(f"Generation failed for {scene}: {e}")
            self.stage_stats["generation"]["attempted"] += num_cases
            self.stage_stats["generation"]["failed"] += num_cases
            # Create failed placeholders
            for i in range(num_cases):
                results.append(
                    CaseResult(
                        case_id=f"{scene}_{i:03d}",
                        scene=scene,
                        query="",
                        generation=StageResult(success=False, error=str(e)),
                        status="failed",
                        failure_stage="generation",
                        failure_reason=str(e),
                    )
                )

        return results

    def _run_parser(self, case_result: CaseResult, selector) -> StageResult:
        """Run QueryParser on a case using the cached selector."""
        start = time.time()

        try:
            # Use parse_query_hypotheses (correct API)
            hypothesis_output = selector.parse_query_hypotheses(case_result.query)

            return StageResult(
                success=True,
                data={
                    "hypothesis_output": hypothesis_output.model_dump() if hasattr(hypothesis_output, "model_dump") else str(hypothesis_output),
                    "num_hypotheses": len(hypothesis_output.hypotheses) if hasattr(hypothesis_output, "hypotheses") else 0,
                },
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            return StageResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start,
            )

    def _run_selector(self, case_result: CaseResult, selector, parse_data: Dict) -> StageResult:
        """Run KeyframeSelector on a case using the cached selector."""
        start = time.time()

        try:
            result = selector.select_keyframes_v2(case_result.query, k=3)

            return StageResult(
                success=True,
                data={
                    "keyframe_paths": [str(p) for p in result.keyframe_paths],
                    "keyframe_indices": result.keyframe_indices,
                    "target_objects": [
                        {"obj_id": obj.obj_id, "category": obj.category}
                        for obj in result.target_objects
                    ],
                    "matched_obj_ids": [obj.obj_id for obj in result.target_objects],
                },
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            return StageResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start,
            )

    def _run_evaluator(self, case_result: CaseResult) -> StageResult:
        """Run LLMEvaluatorV2 on a case."""
        from conceptgraph.query_scene.llm_evaluator_v2 import (
            EvaluationInputV2,
            get_scene_bev_path,
        )

        start = time.time()
        scene_path = self.replica_root / case_result.scene

        try:
            gen_data = case_result.generation.data
            parse_data = case_result.parsing.data
            select_data = case_result.selection.data

            # Get BEV path
            bev_path = get_scene_bev_path(scene_path)

            # Select best hypothesis using strategy
            hypothesis_output = parse_data.get("hypothesis_output", {})
            hypotheses = hypothesis_output.get("hypotheses", [])
            best_hypo, hypo_rank, hypo_kind = self._select_best_hypothesis(hypotheses)

            # Extract parsed info from selected hypothesis
            parsed_targets, parsed_anchors, parsed_relation = self._extract_from_hypothesis(best_hypo)

            # Build GT frame path (join with scene_path)
            gt_frame_path = None
            if gen_data.get("source_frame_path"):
                gt_frame_path = scene_path / gen_data["source_frame_path"]

            # Build evaluation input
            eval_input = EvaluationInputV2(
                query=case_result.query,
                # GT from generation
                gt_target_obj_ids=gen_data.get("target_obj_ids", []),
                gt_target_categories=gen_data.get("target_categories", []),
                gt_anchor_categories=gen_data.get("anchor_categories", []),
                gt_spatial_relation=gen_data.get("spatial_relation"),
                gt_source_view_id=gen_data.get("source_view_id", -1),
                gt_source_frame_path=gt_frame_path,
                # Parsed from parser (using selected hypothesis)
                parsed_target_categories=parsed_targets,
                parsed_anchor_categories=parsed_anchors,
                parsed_spatial_relation=parsed_relation,
                hypothesis_kind=hypo_kind,
                hypothesis_rank=hypo_rank,
                raw_hypothesis_json=json.dumps(hypothesis_output),
                # Selected from selector
                selected_keyframe_paths=[Path(p) for p in select_data.get("keyframe_paths", [])],
                selected_view_ids=select_data.get("keyframe_indices", []),
                matched_obj_ids=select_data.get("matched_obj_ids", []),
                # BEV
                bev_image_path=bev_path,
                enable_diagnostic_mode=True,
            )

            # Run evaluator with retry wrapper
            evaluator = self._get_evaluator()
            result = evaluator._evaluate_with_retry(eval_input)

            return StageResult(
                success=result.evaluation_status.value == "success",
                data={
                    "parse_score": result.parse_metrics.parse_score,
                    "selector_score": result.selector_evaluation.selector_score,
                    "overall_score": result.overall_score,
                    "can_answer_query": result.selector_evaluation.can_answer_query,
                    "gt_coverage": result.gt_comparison.coverage if result.gt_comparison else None,
                    "suggestions": result.suggestions,
                    "issues": result.selector_evaluation.issues,
                },
                error=result.error_message,
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            return StageResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start,
            )

    def _select_best_hypothesis(self, hypotheses: List[Dict]) -> Tuple[Optional[Dict], int, str]:
        """Select the best hypothesis using priority strategy: direct > proxy > context."""
        if not hypotheses:
            return None, 0, "unknown"

        priority = {"direct": 0, "proxy": 1, "context": 2}

        def sort_key(h: Dict) -> Tuple[int, int]:
            kind = h.get("kind", "context")
            if isinstance(kind, dict):
                kind = kind.get("value", "context")
            rank = h.get("rank", 99) or 99
            return (priority.get(str(kind).lower(), 99), rank)

        sorted_hypos = sorted(hypotheses, key=sort_key)
        best = sorted_hypos[0]

        kind = best.get("kind", "context")
        if isinstance(kind, dict):
            kind = kind.get("value", "context")

        original_rank = best.get("rank", 1) or 1

        return best, original_rank, str(kind).lower()

    def _extract_from_hypothesis(self, hypo: Optional[Dict]) -> Tuple[List[str], List[str], Optional[str]]:
        """Extract target categories, anchor categories, and spatial relation from hypothesis.

        Schema: hypothesis.grounding_query.root is a QueryNode with:
        - categories: List[str]
        - spatial_constraints: List[SpatialConstraint]
          - SpatialConstraint.relation: str
          - SpatialConstraint.anchors: List[QueryNode]
        """
        if not hypo:
            return [], [], None

        try:
            gq = hypo.get("grounding_query", {})
            root = gq.get("root", {})

            # Target categories from root.categories
            target_categories = root.get("categories", [])

            # Anchor categories and relation from spatial_constraints
            anchor_categories = []
            spatial_relation = None

            spatial_constraints = root.get("spatial_constraints", [])
            if spatial_constraints:
                # Take first constraint
                first_constraint = spatial_constraints[0]
                spatial_relation = first_constraint.get("relation")

                # Get anchors from constraint
                anchors = first_constraint.get("anchors", [])
                for anchor in anchors:
                    anchor_categories.extend(anchor.get("categories", []))

            return target_categories, anchor_categories, spatial_relation

        except Exception as e:
            logger.warning(f"Failed to extract from hypothesis: {e}")
            return [], [], None

    def _checkpoint(self, stage: str, name: str, data: Any) -> None:
        """Save checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_file = checkpoint_dir / f"{stage}_{name}.json"

        # Convert to serializable format
        if isinstance(data, list):
            serializable = [asdict(d) if hasattr(d, "__dataclass_fields__") else d for d in data]
        else:
            serializable = data

        with open(checkpoint_file, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.debug(f"Checkpoint saved: {checkpoint_file}")

    def _save_config(self) -> None:
        """Save configuration."""
        config_file = self.output_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    def _generate_report(self) -> Dict[str, Any]:
        """Generate summary report."""
        successful = [r for r in self.results if r.status == "success"]
        failed = [r for r in self.results if r.status == "failed"]

        # Score statistics
        parse_scores = [r.parse_score for r in successful if r.parse_score is not None]
        selector_scores = [r.selector_score for r in successful if r.selector_score is not None]
        overall_scores = [r.overall_score for r in successful if r.overall_score is not None]

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        # Failures by stage
        failures_by_stage = {}
        for r in failed:
            stage = r.failure_stage or "unknown"
            failures_by_stage[stage] = failures_by_stage.get(stage, 0) + 1

        # By scene
        by_scene = {}
        for r in self.results:
            if r.scene not in by_scene:
                by_scene[r.scene] = {"total": 0, "success": 0, "scores": []}
            by_scene[r.scene]["total"] += 1
            if r.status == "success":
                by_scene[r.scene]["success"] += 1
                if r.overall_score is not None:
                    by_scene[r.scene]["scores"].append(r.overall_score)

        for scene in by_scene:
            scores = by_scene[scene]["scores"]
            by_scene[scene]["avg_score"] = avg(scores)
            del by_scene[scene]["scores"]

        return {
            "config": asdict(self.config),
            "summary": {
                "total_cases": len(self.results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(self.results) if self.results else 0,
            },
            "scores": {
                "avg_parse_score": round(avg(parse_scores), 2),
                "avg_selector_score": round(avg(selector_scores), 2),
                "avg_overall_score": round(avg(overall_scores), 2),
                "min_overall": round(min(overall_scores), 2) if overall_scores else None,
                "max_overall": round(max(overall_scores), 2) if overall_scores else None,
            },
            "stage_stats": self.stage_stats,
            "failures_by_stage": failures_by_stage,
            "by_scene": by_scene,
            "timestamp": datetime.now().isoformat(),
        }

    def _save_results(self, report: Dict[str, Any]) -> None:
        """Save all results."""
        # Save report
        report_file = self.output_dir / "report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved: {report_file}")

        # Save detailed results
        results_file = self.output_dir / "results.json"
        results_data = [asdict(r) for r in self.results]
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2, default=str)
        logger.info(f"Results saved: {results_file}")

        # Print summary
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total cases: {report['summary']['total_cases']}")
        logger.info(f"Successful: {report['summary']['successful']}")
        logger.info(f"Failed: {report['summary']['failed']}")
        logger.info(f"Success rate: {report['summary']['success_rate']:.1%}")
        logger.info(f"Avg parse score: {report['scores']['avg_parse_score']}")
        logger.info(f"Avg selector score: {report['scores']['avg_selector_score']}")
        logger.info(f"Avg overall score: {report['scores']['avg_overall_score']}")
        logger.info(f"Stage stats: {report['stage_stats']}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="E2E evaluation pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pilot", "smoke", "full"],
        default="pilot",
        help="Execution mode (default: pilot)",
    )
    parser.add_argument(
        "--replica_root",
        type=str,
        default="/Users/bytedance/Replica",
        help="Path to Replica dataset root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated)",
    )

    args = parser.parse_args()

    replica_root = Path(args.replica_root)
    if not replica_root.exists():
        logger.error(f"Replica root not found: {replica_root}")
        sys.exit(1)

    # Create config
    config = create_config(args.mode, replica_root)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = replica_root / "eval_runs" / f"{config.mode}_{config.run_id}"

    # Run pipeline
    pipeline = E2EPipeline(config, replica_root, output_dir)
    report = pipeline.run()

    logger.info(f"Evaluation complete. Results in: {output_dir}")


if __name__ == "__main__":
    main()
