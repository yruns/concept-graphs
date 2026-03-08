"""
Tests for QueryParser outputting HypothesisOutputV1.

These tests verify that:
1. The parser returns HypothesisOutputV1 (not GroundingQuery)
2. parse_mode decisions are correct (SINGLE vs MULTI)
3. Hypothesis kinds are generated correctly (DIRECT, PROXY, CONTEXT)
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch, MagicMock

from conceptgraph.query_scene.query_parser import QueryParser
from conceptgraph.query_scene.query_structures import (
    HypothesisOutputV1,
    ParseMode,
    HypothesisKind,
)


class TestQueryParserHypothesisOutput(unittest.TestCase):
    """Test that QueryParser returns HypothesisOutputV1."""

    def _mock_llm_response(self, response_json: str):
        """Create a mock LLM that returns the given JSON."""
        mock_response = Mock()
        mock_response.content = response_json
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    @patch.object(QueryParser, '_get_fresh_llm')
    def test_parse_returns_hypothesis_output_v1(self, mock_get_llm):
        """Verify parse() returns HypothesisOutputV1, not GroundingQuery."""
        response_json = '''
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the pillow on the sofa",
        "root": {
          "categories": ["pillow"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["pillow", "sofa"]
    }
  ]
}
'''
        mock_get_llm.return_value = self._mock_llm_response(response_json)

        parser = QueryParser(
            llm_model="gemini-test",
            scene_categories=["pillow", "sofa", "door"],
        )
        result = parser.parse("the pillow on the sofa")

        self.assertIsInstance(result, HypothesisOutputV1)
        self.assertEqual(result.parse_mode, ParseMode.SINGLE)
        self.assertEqual(len(result.hypotheses), 1)
        self.assertEqual(result.hypotheses[0].kind, HypothesisKind.DIRECT)

    @patch.object(QueryParser, '_get_fresh_llm')
    def test_parse_simple_returns_single_mode(self, mock_get_llm):
        """When target and anchor exist, should return SINGLE mode with one DIRECT hypothesis."""
        response_json = '''
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the pillow on the sofa",
        "root": {
          "categories": ["pillow", "throw_pillow"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["pillow", "sofa"]
    }
  ]
}
'''
        mock_get_llm.return_value = self._mock_llm_response(response_json)

        parser = QueryParser(
            llm_model="gemini-test",
            scene_categories=["pillow", "throw_pillow", "sofa", "door"],
        )
        result = parser.parse("the pillow on the sofa")

        self.assertEqual(result.parse_mode, ParseMode.SINGLE)
        self.assertEqual(len(result.hypotheses), 1)
        self.assertEqual(result.hypotheses[0].kind, HypothesisKind.DIRECT)
        # Check semantic expansion
        self.assertIn("pillow", result.hypotheses[0].grounding_query.root.categories)
        self.assertIn("throw_pillow", result.hypotheses[0].grounding_query.root.categories)

    @patch.object(QueryParser, '_get_fresh_llm')
    def test_parse_missing_anchor_returns_multi_mode(self, mock_get_llm):
        """When anchor is missing (UNKNOW), should return MULTI mode with DIRECT + PROXY."""
        response_json = '''
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "multi",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the pillow on the bed",
        "root": {
          "categories": ["pillow", "throw_pillow"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["UNKNOW"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["pillow", "bed"]
    },
    {
      "kind": "proxy",
      "rank": 2,
      "grounding_query": {
        "raw_query": "proxy for: the pillow on the bed",
        "root": {
          "categories": ["pillow", "throw_pillow"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["proxy_anchor"]
    }
  ]
}
'''
        mock_get_llm.return_value = self._mock_llm_response(response_json)

        parser = QueryParser(
            llm_model="gemini-test",
            scene_categories=["pillow", "throw_pillow", "sofa", "door"],  # No "bed"
        )
        result = parser.parse("the pillow on the bed")

        self.assertEqual(result.parse_mode, ParseMode.MULTI)
        self.assertEqual(len(result.hypotheses), 2)
        # First hypothesis is DIRECT with UNKNOW anchor
        self.assertEqual(result.hypotheses[0].kind, HypothesisKind.DIRECT)
        self.assertIn("UNKNOW", result.hypotheses[0].grounding_query.root.spatial_constraints[0].anchors[0].categories)
        # Second hypothesis is PROXY with sofa as proxy anchor
        self.assertEqual(result.hypotheses[1].kind, HypothesisKind.PROXY)
        self.assertIn("sofa", result.hypotheses[1].grounding_query.root.spatial_constraints[0].anchors[0].categories)

    @patch.object(QueryParser, '_get_fresh_llm')
    def test_parse_missing_target_returns_multi_with_context(self, mock_get_llm):
        """When target is missing, should return MULTI mode with DIRECT + PROXY + CONTEXT."""
        response_json = '''
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "multi",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the laptop on the table",
        "root": {
          "categories": ["UNKNOW"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["side_table"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["laptop", "table"]
    },
    {
      "kind": "proxy",
      "rank": 2,
      "grounding_query": {
        "raw_query": "proxy for: the laptop on the table",
        "root": {
          "categories": ["book", "cup"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["side_table"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["proxy"]
    },
    {
      "kind": "context",
      "rank": 3,
      "grounding_query": {
        "raw_query": "context for: the laptop on the table",
        "root": {
          "categories": ["side_table"],
          "attributes": [],
          "spatial_constraints": [],
          "select_constraint": null
        },
        "expect_unique": false
      },
      "lexical_hints": ["context"]
    }
  ]
}
'''
        mock_get_llm.return_value = self._mock_llm_response(response_json)

        parser = QueryParser(
            llm_model="gemini-test",
            scene_categories=["book", "cup", "side_table", "chair"],  # No "laptop"
        )
        result = parser.parse("the laptop on the table")

        self.assertEqual(result.parse_mode, ParseMode.MULTI)
        self.assertEqual(len(result.hypotheses), 3)
        # Check kinds
        kinds = [h.kind for h in result.hypotheses]
        self.assertEqual(kinds, [HypothesisKind.DIRECT, HypothesisKind.PROXY, HypothesisKind.CONTEXT])
        # Check ranks
        ranks = [h.rank for h in result.hypotheses]
        self.assertEqual(ranks, [1, 2, 3])
        # CONTEXT should have no spatial constraints
        context_hypo = result.hypotheses[2]
        self.assertEqual(len(context_hypo.grounding_query.root.spatial_constraints), 0)
        self.assertEqual(context_hypo.grounding_query.expect_unique, False)

    @patch.object(QueryParser, '_get_fresh_llm')
    def test_parse_semantic_expansion_stays_single(self, mock_get_llm):
        """Semantic expansion (cushion -> pillow, sofa_seat_cushion) should stay SINGLE mode."""
        response_json = '''
{
  "format_version": "hypothesis_output_v1",
  "parse_mode": "single",
  "hypotheses": [
    {
      "kind": "direct",
      "rank": 1,
      "grounding_query": {
        "raw_query": "the cushion on the couch",
        "root": {
          "categories": ["sofa_seat_cushion", "pillow", "throw_pillow"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null}]
            }
          ],
          "select_constraint": null
        },
        "expect_unique": true
      },
      "lexical_hints": ["cushion", "couch"]
    }
  ]
}
'''
        mock_get_llm.return_value = self._mock_llm_response(response_json)

        parser = QueryParser(
            llm_model="gemini-test",
            scene_categories=["sofa", "sofa_seat_cushion", "pillow", "throw_pillow", "door"],
        )
        result = parser.parse("the cushion on the couch")

        self.assertEqual(result.parse_mode, ParseMode.SINGLE)
        self.assertEqual(len(result.hypotheses), 1)
        # Check semantic expansion
        categories = result.hypotheses[0].grounding_query.root.categories
        self.assertIn("sofa_seat_cushion", categories)
        self.assertIn("pillow", categories)
        self.assertIn("throw_pillow", categories)

    def test_hypothesis_output_validation(self):
        """Test that HypothesisOutputV1 validation rules are enforced."""
        from conceptgraph.query_scene.query_structures import (
            QueryHypothesis,
            GroundingQuery,
            QueryNode,
        )

        # Valid SINGLE mode
        valid_single = HypothesisOutputV1(
            parse_mode=ParseMode.SINGLE,
            hypotheses=[
                QueryHypothesis(
                    kind=HypothesisKind.DIRECT,
                    rank=1,
                    grounding_query=GroundingQuery(
                        raw_query="test",
                        root=QueryNode(categories=["pillow"]),
                    ),
                )
            ],
        )
        self.assertEqual(valid_single.parse_mode, ParseMode.SINGLE)

        # Invalid: SINGLE mode with multiple hypotheses
        with self.assertRaises(ValueError):
            HypothesisOutputV1(
                parse_mode=ParseMode.SINGLE,
                hypotheses=[
                    QueryHypothesis(
                        kind=HypothesisKind.DIRECT,
                        rank=1,
                        grounding_query=GroundingQuery(
                            raw_query="test",
                            root=QueryNode(categories=["pillow"]),
                        ),
                    ),
                    QueryHypothesis(
                        kind=HypothesisKind.PROXY,
                        rank=2,
                        grounding_query=GroundingQuery(
                            raw_query="test",
                            root=QueryNode(categories=["sofa"]),
                        ),
                    ),
                ],
            )

        # Invalid: SINGLE mode with non-DIRECT kind
        with self.assertRaises(ValueError):
            HypothesisOutputV1(
                parse_mode=ParseMode.SINGLE,
                hypotheses=[
                    QueryHypothesis(
                        kind=HypothesisKind.PROXY,
                        rank=1,
                        grounding_query=GroundingQuery(
                            raw_query="test",
                            root=QueryNode(categories=["pillow"]),
                        ),
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
