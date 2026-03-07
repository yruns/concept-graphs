from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as jsonschema_validate
from pydantic import ValidationError as PydanticValidationError

from conceptgraph.query_scene.query_structures import HypothesisOutputV1


SCENE_CATEGORIES = [
    "pillow",
    "sofa",
    "door",
    "armchair",
    "side_table",
]
SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "schema" / "hypothesis_output_v1.json"
)
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _node(categories: list[str], relation: str | None = None, anchor: str | None = None) -> dict:
    spatial_constraints = []
    if relation and anchor:
        spatial_constraints = [
            {
                "relation": relation,
                "anchors": [
                    {
                        "categories": [anchor],
                        "attributes": [],
                        "spatial_constraints": [],
                        "select_constraint": None,
                        "node_id": "anchor",
                    }
                ],
            }
        ]

    return {
        "categories": categories,
        "attributes": [],
        "spatial_constraints": spatial_constraints,
        "select_constraint": None,
        "node_id": "root",
    }


def _grounding_query(category: str, anchor: str | None = "sofa") -> dict:
    relation = "near" if anchor else None
    return {
        "raw_query": f"find {category}",
        "root": _node([category], relation=relation, anchor=anchor),
        "expect_unique": category != "UNKNOW",
    }


def _single_payload(i: int) -> dict:
    category = "pillow" if i % 2 == 0 else "side_table"
    return {
        "format_version": "hypothesis_output_v1",
        "parse_mode": "single",
        "hypotheses": [
            {
                "kind": "direct",
                "rank": 1,
                "grounding_query": _grounding_query(category),
                "lexical_hints": ["hint", str(i)],
            }
        ],
    }


def _multi_payload(i: int) -> dict:
    return {
        "format_version": "hypothesis_output_v1",
        "parse_mode": "multi",
        "hypotheses": [
            {
                "kind": "direct",
                "rank": 1,
                "grounding_query": _grounding_query("UNKNOW"),
                "lexical_hints": ["unknown", str(i)],
            },
            {
                "kind": "proxy",
                "rank": 2,
                "grounding_query": _grounding_query("pillow"),
                "lexical_hints": ["cushion"],
            },
            {
                "kind": "context",
                "rank": 3,
                "grounding_query": _grounding_query("sofa", anchor="door"),
                "lexical_hints": ["context"],
            },
        ],
    }


def _validate_payload(payload: dict, hidden_categories: set[str] | None = None) -> None:
    jsonschema_validate(instance=payload, schema=SCHEMA)
    parsed = HypothesisOutputV1.model_validate(payload)
    parsed.validate_categories(SCENE_CATEGORIES)
    if hidden_categories:
        parsed.validate_no_mask_leak(hidden_categories)


class TestHypothesisOutputSchema(unittest.TestCase):
    def test_schema_positive_examples_20_pass(self) -> None:
        payloads = [_single_payload(i) for i in range(10)] + [_multi_payload(i) for i in range(10)]
        self.assertEqual(len(payloads), 20)

        for payload in payloads:
            _validate_payload(payload)

    def test_schema_negative_examples_20_rejected(self) -> None:
        base_single = _single_payload(0)
        base_multi = _multi_payload(0)

        invalid_cases: list[tuple[dict, set[str] | None]] = []

        p = deepcopy(base_single)
        p["format_version"] = "hypothesis_output_v0"
        invalid_cases.append((p, None))

        p = deepcopy(base_single)
        p["parse_mode"] = "parallel"
        invalid_cases.append((p, None))

        p = deepcopy(base_single)
        p["hypotheses"] = []
        invalid_cases.append((p, None))

        p = deepcopy(base_multi)
        p["hypotheses"].append(deepcopy(p["hypotheses"][0]))
        invalid_cases.append((p, None))

        p = deepcopy(base_single)
        del p["hypotheses"][0]["kind"]
        invalid_cases.append((p, None))

        p = deepcopy(base_single)
        p["hypotheses"][0]["kind"] = "hard"
        invalid_cases.append((p, None))

        p = deepcopy(base_single)
        p["hypotheses"][0]["rank"] = 0
        invalid_cases.append((p, None))

        p = deepcopy(base_multi)
        p["hypotheses"][1]["rank"] = 1
        invalid_cases.append((p, None))

        p = deepcopy(base_multi)
        p["hypotheses"][1]["rank"] = 3
        p["hypotheses"][2]["rank"] = 4
        invalid_cases.append((p, None))

        p = deepcopy(base_single)
        p["hypotheses"].append(deepcopy(p["hypotheses"][0]))
        invalid_cases.append((p, None))

        p = deepcopy(base_single)
        p["hypotheses"][0]["kind"] = "proxy"
        invalid_cases.append((p, None))

        p = deepcopy(base_single)
        p["hypotheses"][0]["grounding_query"]["root"]["categories"] = ["cushion"]
        invalid_cases.append((p, None))

        p = deepcopy(base_single)
        p["hypotheses"][0]["lexical_hints"] = "hint"
        invalid_cases.append((p, None))

        p = deepcopy(base_single)
        del p["hypotheses"][0]["grounding_query"]["root"]
        invalid_cases.append((p, None))

        p = deepcopy(base_single)
        p["hypotheses"][0]["grounding_query"]["root"]["categories"] = []
        invalid_cases.append((p, None))

        p = deepcopy(base_single)
        p["hypotheses"][0]["grounding_query"]["root"]["spatial_constraints"] = [
            {"relation": "near", "anchors": []}
        ]
        invalid_cases.append((p, None))

        p = deepcopy(base_single)
        p["hypotheses"][0]["grounding_query"]["root"]["select_constraint"] = {
            "constraint_type": "ordinal",
            "metric": "x_position",
            "order": "asc",
            "reference": None,
            "position": None,
        }
        invalid_cases.append((p, None))

        p = deepcopy(base_single)
        del p["hypotheses"][0]["grounding_query"]["root"]["node_id"]
        invalid_cases.append((p, None))

        p = deepcopy(base_single)
        p["hypotheses"][0]["grounding_query"]["root"]["unexpected"] = "x"
        invalid_cases.append((p, None))

        p = deepcopy(base_single)
        p["hypotheses"][0]["grounding_query"]["root"]["categories"] = ["pillow"]
        invalid_cases.append((p, {"pillow"}))

        self.assertEqual(len(invalid_cases), 20)

        for payload, hidden_categories in invalid_cases:
            with self.assertRaises((JsonSchemaValidationError, PydanticValidationError, ValueError, TypeError)):
                _validate_payload(payload, hidden_categories=hidden_categories)


if __name__ == "__main__":
    unittest.main()
