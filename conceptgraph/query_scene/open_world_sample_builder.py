"""Build parser_sft and retrieval_eval samples from scene/program assets."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .query_structures import HypothesisOutputV1


TERM_SYNONYMS = {
    "pillow": ["cushion"],
    "throw_pillow": ["throw cushion", "cushion"],
    "sofa": ["couch"],
    "armchair": ["lounge chair"],
    "side_table": ["end table"],
    "door": ["entryway"],
    "floor_lamp": ["lamp"],
    "wall_sconce": ["wall lamp"],
    "ottoman": ["footstool"],
}

RELATION_TEXT = {
    "near": "near",
    "on": "on",
    "next_to": "next to",
    "above": "above",
    "below": "below",
}

DEFAULT_TEACHER_MODELS = [
    "gpt-5.2-2025-12-11",
    "gemini-3-pro-preview-new",
]
DEFAULT_PROMPT_VERSION = "p_qwen_sft_v3_20260307"


def _stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


class TeacherQueryGenerator:
    """Dual-teacher query generation with persistent cache and retries."""

    def __init__(
        self,
        cache_path: Path,
        teacher_models: Sequence[str],
        prompt_version: str,
        temperature: float,
        seed: int,
        max_retries: int = 2,
        llm_factory: Optional[Callable[..., Any]] = None,
    ):
        self.cache_path = cache_path
        self.teacher_models = list(teacher_models)
        self.prompt_version = prompt_version
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries

        if llm_factory is None:
            from conceptgraph.utils.llm_client import get_langchain_chat_model

            self.llm_factory = get_langchain_chat_model
        else:
            self.llm_factory = llm_factory

        self._clients: Dict[str, Any] = {}
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.new_entries: List[Dict[str, Any]] = []
        self.failure_entries: List[Dict[str, Any]] = []

        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return

        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = record.get("key")
                if key:
                    self.cache[key] = record

    def _append_cache(self, record: Dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _cache_key(self, scene_id: str, program_hash: str, prompt_hash: str, model: str) -> str:
        # As required by plan: scene_id + program_hash + prompt_hash + model
        return f"{scene_id}::{program_hash}::{prompt_hash}::{model}"

    def _get_client(self, model: str) -> Any:
        if model not in self._clients:
            self._clients[model] = self.llm_factory(
                deployment_name=model,
                temperature=self.temperature,
                timeout=120,
                max_retries=2,
            )
        return self._clients[model]

    def _invoke_model(self, model: str, prompt: str) -> str:
        client = self._get_client(model)
        response = client.invoke(prompt)
        content = getattr(response, "content", response)

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", item)))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)

    def _extract_query(self, raw_text: str) -> str:
        text = raw_text.strip()
        if not text:
            raise ValueError("Empty teacher response")

        # Strip markdown code fences.
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()

        # Try JSON style response first.
        json_candidate = text
        if text.startswith("{") and text.endswith("}"):
            try:
                payload = json.loads(text)
                for key in ["query", "user_query", "text"]:
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        text = value.strip()
                        break
            except json.JSONDecodeError:
                pass
        else:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                json_candidate = match.group(0)
                try:
                    payload = json.loads(json_candidate)
                    for key in ["query", "user_query", "text"]:
                        value = payload.get(key)
                        if isinstance(value, str) and value.strip():
                            text = value.strip()
                            break
                except json.JSONDecodeError:
                    pass

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            text = lines[0]

        text = text.strip().strip('"').strip("'").strip("`")
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 3:
            raise ValueError("Teacher query too short")
        return text

    def _generate_with_retry(
        self,
        scene_id: str,
        program_hash: str,
        model: str,
        prompt: str,
        prompt_hash: str,
    ) -> Dict[str, Any]:
        last_error = ""
        raw_response = ""

        for attempt in range(1, self.max_retries + 1):
            try:
                raw_response = self._invoke_model(model, prompt)
                query = self._extract_query(raw_response)
                return {
                    "key": self._cache_key(scene_id, program_hash, prompt_hash, model),
                    "scene_id": scene_id,
                    "program_hash": program_hash,
                    "model": model,
                    "prompt_hash": prompt_hash,
                    "prompt_version": self.prompt_version,
                    "temperature": self.temperature,
                    "seed": self.seed,
                    "status": "success",
                    "query": query,
                    "error": None,
                    "attempts": attempt,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

        return {
            "key": self._cache_key(scene_id, program_hash, prompt_hash, model),
            "scene_id": scene_id,
            "program_hash": program_hash,
            "model": model,
            "prompt_hash": prompt_hash,
            "prompt_version": self.prompt_version,
            "temperature": self.temperature,
            "seed": self.seed,
            "status": "failure",
            "query": None,
            "error": last_error,
            "attempts": self.max_retries,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def generate_for_sample(
        self,
        scene_id: str,
        program_hash: str,
        prompt: str,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        prompt_hash = _stable_hash(f"{self.prompt_version}\n{prompt}", length=16)

        candidates: List[Dict[str, Any]] = []
        successes: List[Dict[str, Any]] = []

        for model in self.teacher_models:
            key = self._cache_key(scene_id, program_hash, prompt_hash, model)
            if key in self.cache:
                entry = deepcopy(self.cache[key])
                entry["cache_hit"] = True
            else:
                entry = self._generate_with_retry(
                    scene_id=scene_id,
                    program_hash=program_hash,
                    model=model,
                    prompt=prompt,
                    prompt_hash=prompt_hash,
                )
                entry["cache_hit"] = False
                self.cache[key] = deepcopy(entry)
                self._append_cache(entry)
                self.new_entries.append(deepcopy(entry))
                if entry.get("status") == "failure":
                    self.failure_entries.append(deepcopy(entry))

            candidate = {
                "model": model,
                "status": entry.get("status"),
                "query": entry.get("query"),
                "error": entry.get("error"),
                "cache_hit": bool(entry.get("cache_hit", False)),
                "attempts": entry.get("attempts"),
            }
            candidates.append(candidate)
            if entry.get("status") == "success" and entry.get("query"):
                successes.append(entry)

        selected_query: Optional[str] = None
        selected_model: Optional[str] = None
        selected_from_cache = False

        if successes:
            selector_text = f"{scene_id}:{program_hash}:{prompt_hash}:{self.seed}"
            idx = int(hashlib.sha1(selector_text.encode("utf-8")).hexdigest(), 16) % len(successes)
            chosen = successes[idx]
            selected_query = chosen["query"]
            selected_model = chosen["model"]
            selected_from_cache = bool(chosen.get("cache_hit", False))

        metadata = {
            "prompt_version": self.prompt_version,
            "prompt_hash": prompt_hash,
            "temperature": self.temperature,
            "seed": self.seed,
            "selected_model": selected_model,
            "selected_from_cache": selected_from_cache,
            "all_failed": selected_query is None,
            "candidates": candidates,
        }
        return selected_query, metadata

    def stats(self) -> Dict[str, Any]:
        success_new = sum(1 for rec in self.new_entries if rec.get("status") == "success")
        fail_new = sum(1 for rec in self.new_entries if rec.get("status") == "failure")
        return {
            "teacher_models": self.teacher_models,
            "prompt_version": self.prompt_version,
            "temperature": self.temperature,
            "seed": self.seed,
            "cache_path": str(self.cache_path),
            "cache_total_entries": len(self.cache),
            "new_cache_entries": len(self.new_entries),
            "new_success_entries": success_new,
            "new_failure_entries": fail_new,
        }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def bucket_counts(total: int) -> Dict[str, int]:
    direct = int(total * 0.4)
    soft = int(total * 0.3)
    hard = total - direct - soft
    return {"direct": direct, "soft": soft, "hard": hard}


def _sample_with_replacement(records: Sequence[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    if not records:
        return []
    if n <= len(records):
        return rng.sample(list(records), n)

    out: List[Dict[str, Any]] = []
    pool = list(records)
    while len(out) < n:
        shuffled = pool[:]
        rng.shuffle(shuffled)
        out.extend(shuffled)
    return out[:n]


def _canonical_phrase(category: str) -> str:
    return category.replace("_", " ")


def _soft_phrase(category: str, rng: random.Random) -> str:
    options = TERM_SYNONYMS.get(category, [])
    if options:
        return rng.choice(options)
    return _canonical_phrase(category)


def _render_user_query(program: Dict[str, Any], bucket: str, rng: random.Random) -> str:
    target = program["target_category"]
    anchor = program["anchor_categories"][0] if program.get("anchor_categories") else None
    relation = program.get("relation")
    program_type = program["program_type"]

    if bucket == "direct":
        target_text = _canonical_phrase(target)
        anchor_text = _canonical_phrase(anchor) if anchor else None
    else:
        target_text = _soft_phrase(target, rng)
        anchor_text = _soft_phrase(anchor, rng) if anchor else None

    if program_type == "superlative" and anchor_text:
        return f"find the {target_text} nearest the {anchor_text}"

    if relation and anchor_text:
        rel_text = RELATION_TEXT.get(relation, relation.replace("_", " "))
        return f"find the {target_text} {rel_text} the {anchor_text}"

    return f"find the {target_text}"


def _choose_hidden_categories(target: str, scene_categories: Sequence[str]) -> List[str]:
    hidden = []
    for cat in scene_categories:
        if cat == target or target in cat or cat in target:
            hidden.append(cat)
    if target not in hidden:
        hidden.append(target)
    return sorted(set(hidden))


def _build_teacher_prompt(
    *,
    bucket: str,
    scene_categories_visible: Sequence[str],
    hidden_categories: Sequence[str],
    grounding_query: Dict[str, Any],
    prompt_version: str,
) -> str:
    hidden_text = ", ".join(hidden_categories) if hidden_categories else "none"
    scene_text = ", ".join(scene_categories_visible)
    grounding_json = json.dumps(grounding_query, ensure_ascii=False)

    extra_hard_rules = ""
    if bucket == "hard":
        extra_hard_rules = (
            "- This is a HARD case: parser cannot directly see hidden categories. "
            "The query should still naturally refer to the hidden target concept.\n"
            "- Do not mention system tokens like UNKNOW, proxy, context.\n"
        )

    prompt = (
        "You generate realistic user queries for 3D scene retrieval training.\n"
        f"PROMPT_VERSION: {prompt_version}\n"
        f"BUCKET: {bucket}\n"
        f"VISIBLE_SCENE_CATEGORIES: [{scene_text}]\n"
        f"HIDDEN_CATEGORIES: [{hidden_text}]\n"
        f"GROUNDING_INTENT_JSON: {grounding_json}\n\n"
        "Write exactly ONE natural English query that matches the grounding intent.\n"
        "Rules:\n"
        "- Keep the semantic intent unchanged (target/relation/anchor).\n"
        "- Prefer natural wording and paraphrases/synonyms where appropriate.\n"
        "- Keep length between 6 and 24 words.\n"
        "- Output plain text only. No JSON, no markdown, no explanation.\n"
        f"{extra_hard_rules}"
    )
    return prompt


def _build_single_output(
    user_query: str,
    grounding_query: Dict[str, Any],
    lexical_hints: List[str],
) -> Dict[str, Any]:
    gq = deepcopy(grounding_query)
    gq["raw_query"] = user_query
    return {
        "format_version": "hypothesis_output_v1",
        "parse_mode": "single",
        "hypotheses": [
            {
                "kind": "direct",
                "rank": 1,
                "grounding_query": gq,
                "lexical_hints": lexical_hints,
            }
        ],
    }


def _build_hard_output(
    user_query: str,
    grounding_query: Dict[str, Any],
    scene_categories_masked: Sequence[str],
    hidden_categories: Sequence[str],
) -> Tuple[Dict[str, Any], List[str]]:
    base = deepcopy(grounding_query)
    base["raw_query"] = user_query

    direct_gq = deepcopy(base)
    direct_gq["root"]["categories"] = ["UNKNOW"]

    anchor_categories: List[str] = []
    for sc in base["root"].get("spatial_constraints", []):
        for anchor in sc.get("anchors", []):
            anchor_categories.extend(anchor.get("categories", []))

    support_categories = [c for c in anchor_categories if c in scene_categories_masked]
    if not support_categories:
        support_categories = [c for c in scene_categories_masked if c not in hidden_categories]
    if not support_categories:
        support_categories = ["UNKNOW"]

    proxy_gq = deepcopy(base)
    proxy_gq["root"]["categories"] = [support_categories[0]]

    context_gq = deepcopy(base)
    context_gq["root"]["categories"] = support_categories[:2]
    context_gq["root"]["spatial_constraints"] = []
    context_gq["root"]["select_constraint"] = None
    context_gq["expect_unique"] = False

    output = {
        "format_version": "hypothesis_output_v1",
        "parse_mode": "multi",
        "hypotheses": [
            {
                "kind": "direct",
                "rank": 1,
                "grounding_query": direct_gq,
                "lexical_hints": ["unknown_target"],
            },
            {
                "kind": "proxy",
                "rank": 2,
                "grounding_query": proxy_gq,
                "lexical_hints": ["proxy"],
            },
            {
                "kind": "context",
                "rank": 3,
                "grounding_query": context_gq,
                "lexical_hints": ["context"],
            },
        ],
    }
    return output, support_categories


def _validate_output(
    output: Dict[str, Any],
    scene_categories: Sequence[str],
    hidden_categories: Sequence[str] | None = None,
) -> None:
    parsed = HypothesisOutputV1.model_validate(output)
    parsed.validate_categories(scene_categories)
    if hidden_categories:
        parsed.validate_no_mask_leak(hidden_categories)


def build_samples_for_scene(
    scene_manifest: Dict[str, Any],
    programs: Sequence[Dict[str, Any]],
    samples_per_scene: int,
    seed: int = 42,
    teacher_generator: Optional[TeacherQueryGenerator] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    rng = random.Random(seed)
    scene_id = scene_manifest["scene_id"]
    scene_categories = list(scene_manifest["scene_categories"])

    counts = bucket_counts(samples_per_scene)
    hard_candidates = [p for p in programs if p.get("anchor_categories")]
    if not hard_candidates:
        hard_candidates = list(programs)

    bucket_programs = {
        "direct": _sample_with_replacement(programs, counts["direct"], rng),
        "soft": _sample_with_replacement(programs, counts["soft"], rng),
        "hard": _sample_with_replacement(hard_candidates, counts["hard"], rng),
    }

    parser_sft: List[Dict[str, Any]] = []
    retrieval_eval: List[Dict[str, Any]] = []

    for bucket in ["direct", "soft", "hard"]:
        for idx, program in enumerate(bucket_programs[bucket]):
            program_id = program.get("program_id", f"prog_{idx:06d}")
            sample_id = f"{scene_id}_{bucket}_{idx:06d}"
            program_hash = program["program_hash"]

            template_query = _render_user_query(program, bucket=bucket, rng=rng)
            hidden_categories: List[str] = []
            scene_categories_effective = scene_categories

            if bucket == "hard":
                hidden_categories = _choose_hidden_categories(program["target_category"], scene_categories)
                scene_categories_effective = [c for c in scene_categories if c not in hidden_categories]
                if not scene_categories_effective:
                    scene_categories_effective = [c for c in scene_categories if c != program["target_category"]][:1]

            teacher_meta = None
            user_query = template_query
            if teacher_generator is not None:
                teacher_prompt = _build_teacher_prompt(
                    bucket=bucket,
                    scene_categories_visible=scene_categories_effective,
                    hidden_categories=hidden_categories,
                    grounding_query=program["grounding_query"],
                    prompt_version=teacher_generator.prompt_version,
                )
                generated_query, teacher_meta = teacher_generator.generate_for_sample(
                    scene_id=scene_id,
                    program_hash=program_hash,
                    prompt=teacher_prompt,
                )
                if generated_query:
                    user_query = generated_query

            gq = deepcopy(program["grounding_query"])
            gq["raw_query"] = user_query

            if bucket in {"direct", "soft"}:
                hints = [program["target_category"]]
                target_output = _build_single_output(user_query, gq, lexical_hints=hints)
                _validate_output(target_output, scene_categories)

                parser_sft_record = {
                    "sample_id": sample_id,
                    "bucket": bucket,
                    "scene_id": scene_id,
                    "scene_categories": scene_categories,
                    "program_id": program_id,
                    "program_hash": program_hash,
                    "user_query": user_query,
                    "target_output": target_output,
                }
                retrieval_record = {
                    "sample_id": sample_id,
                    "bucket": bucket,
                    "scene_id": scene_id,
                    "mask_spec": {"type": "none", "hidden_categories": []},
                    "scene_categories_masked": scene_categories,
                    "user_query": user_query,
                    "qwen_target_output": target_output,
                    "expected_status": "direct_grounded",
                    "expected_first_hit_kind": "direct",
                    "expected_support_categories": [program["target_category"]] + list(program.get("anchor_categories", [])),
                }
            else:
                target_output, support_categories = _build_hard_output(
                    user_query=user_query,
                    grounding_query=gq,
                    scene_categories_masked=scene_categories_effective,
                    hidden_categories=hidden_categories,
                )
                _validate_output(
                    target_output,
                    scene_categories_effective,
                    hidden_categories=hidden_categories,
                )

                mask_type = "M1+M2" if idx % 2 == 0 else "M1"
                mask_spec = {
                    "type": mask_type,
                    "hidden_categories": hidden_categories,
                }

                parser_sft_record = {
                    "sample_id": sample_id,
                    "bucket": bucket,
                    "scene_id": scene_id,
                    "scene_categories": scene_categories_effective,
                    "mask_spec": mask_spec,
                    "program_id": program_id,
                    "program_hash": program_hash,
                    "user_query": user_query,
                    "target_output": target_output,
                }
                retrieval_record = {
                    "sample_id": sample_id,
                    "bucket": bucket,
                    "scene_id": scene_id,
                    "mask_spec": mask_spec,
                    "scene_categories_masked": scene_categories_effective,
                    "user_query": user_query,
                    "qwen_target_output": target_output,
                    "expected_status": "proxy_grounded",
                    "expected_first_hit_kind": "proxy",
                    "expected_support_categories": support_categories,
                }

            if teacher_meta is not None:
                parser_sft_record["teacher_generation"] = teacher_meta
                retrieval_record["teacher_generation"] = teacher_meta

            # Guarantee retrieval_eval has no gold_keyframes.
            retrieval_record.pop("gold_keyframes", None)
            parser_sft_record.pop("gold_keyframes", None)

            parser_sft.append(parser_sft_record)
            retrieval_eval.append(retrieval_record)

    return parser_sft, retrieval_eval, counts


def build_samples_from_assets(
    scene_manifest_path: Path,
    query_program_pool_path: Path,
    output_dir: Path,
    samples_per_scene: int = 300,
    seed: int = 42,
    use_teacher_llm: bool = False,
    teacher_models: Optional[Sequence[str]] = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    teacher_temperature: float = 0.2,
    teacher_max_retries: int = 2,
    teacher_cache_path: Optional[Path] = None,
    llm_factory: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    scene_manifest = read_jsonl(scene_manifest_path)
    program_pool = read_jsonl(query_program_pool_path)

    programs_by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for rec in program_pool:
        programs_by_scene.setdefault(rec["scene_id"], []).append(rec)

    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_generator = None
    if use_teacher_llm:
        teacher_generator = TeacherQueryGenerator(
            cache_path=teacher_cache_path or (output_dir / "teacher_query_cache.jsonl"),
            teacher_models=teacher_models or DEFAULT_TEACHER_MODELS,
            prompt_version=prompt_version,
            temperature=teacher_temperature,
            seed=seed,
            max_retries=teacher_max_retries,
            llm_factory=llm_factory,
        )

    parser_sft_all: List[Dict[str, Any]] = []
    retrieval_eval_all: List[Dict[str, Any]] = []
    scene_reports = []

    for scene_idx, scene in enumerate(scene_manifest):
        scene_id = scene["scene_id"]
        programs = programs_by_scene.get(scene_id, [])
        if not programs:
            continue

        parser_sft, retrieval_eval, counts = build_samples_for_scene(
            scene_manifest=scene,
            programs=programs,
            samples_per_scene=samples_per_scene,
            seed=seed + scene_idx,
            teacher_generator=teacher_generator,
        )
        parser_sft_all.extend(parser_sft)
        retrieval_eval_all.extend(retrieval_eval)

        scene_reports.append(
            {
                "scene_id": scene_id,
                "samples": len(parser_sft),
                "bucket_counts": counts,
            }
        )

    parser_path = output_dir / "parser_sft.jsonl"
    retrieval_path = output_dir / "retrieval_eval.jsonl"
    report_path = output_dir / "generation_report.md"

    parser_count = write_jsonl(parser_sft_all, parser_path)
    retrieval_count = write_jsonl(retrieval_eval_all, retrieval_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Sample Generation Report\n\n")
        f.write(f"- parser_sft_records: {parser_count}\n")
        f.write(f"- retrieval_eval_records: {retrieval_count}\n")
        f.write(f"- samples_per_scene: {samples_per_scene}\n")
        f.write(f"- seed: {seed}\n")

        if teacher_generator is not None:
            stats = teacher_generator.stats()
            f.write("\n## Teacher Query Generation\n")
            f.write(f"- enabled: true\n")
            f.write(f"- teacher_models: {stats['teacher_models']}\n")
            f.write(f"- prompt_version: {stats['prompt_version']}\n")
            f.write(f"- temperature: {stats['temperature']}\n")
            f.write(f"- cache_path: {stats['cache_path']}\n")
            f.write(f"- cache_total_entries: {stats['cache_total_entries']}\n")
            f.write(f"- new_cache_entries: {stats['new_cache_entries']}\n")
            f.write(f"- new_success_entries: {stats['new_success_entries']}\n")
            f.write(f"- new_failure_entries: {stats['new_failure_entries']}\n")

            if teacher_generator.failure_entries:
                f.write("\n### Teacher Failures\n")
                for failure in teacher_generator.failure_entries:
                    f.write(
                        f"- scene={failure['scene_id']}, program_hash={failure['program_hash']}, "
                        f"model={failure['model']}, error={failure['error']}\n"
                    )
        else:
            f.write("\n## Teacher Query Generation\n")
            f.write("- enabled: false\n")

        f.write("\n## Scene Breakdown\n")
        for rep in scene_reports:
            f.write(
                f"- {rep['scene_id']}: total={rep['samples']}, "
                f"direct={rep['bucket_counts']['direct']}, "
                f"soft={rep['bucket_counts']['soft']}, "
                f"hard={rep['bucket_counts']['hard']}\n"
            )

    summary = {
        "parser_sft_records": parser_count,
        "retrieval_eval_records": retrieval_count,
        "scene_reports": scene_reports,
        "output_dir": str(output_dir.resolve()),
        "use_teacher_llm": use_teacher_llm,
    }
    if teacher_generator is not None:
        summary["teacher_generation"] = teacher_generator.stats()

    return summary
