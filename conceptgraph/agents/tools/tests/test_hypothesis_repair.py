"""Unit tests for the hypothesis_repair Stage-2 tool backend.

Tests cover:
1. Hypothesis inspection and state tracking
2. Switching between direct/proxy/context hypotheses
3. Hypothesis history tracking
4. Fallback order logic
5. Request parsing and action determination
"""

from __future__ import annotations

import pytest

from conceptgraph.agents.models import (
    KeyframeEvidence,
    Stage1HypothesisSummary,
    Stage2EvidenceBundle,
)
from conceptgraph.agents.tools.hypothesis_repair import (
    HypothesisAction,
    HypothesisHistoryEntry,
    HypothesisRepairBackend,
    HypothesisRepairConfig,
    create_hypothesis_repair_callback,
)


@pytest.fixture
def sample_hypotheses():
    """Sample hypotheses in Stage-1 output format."""
    return [
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
                            "anchors": [
                                {
                                    "categories": ["sofa"],
                                    "attributes": [],
                                    "spatial_constraints": [],
                                    "select_constraint": None,
                                }
                            ],
                        }
                    ],
                    "select_constraint": None,
                },
                "expect_unique": True,
            },
            "lexical_hints": ["pillow", "sofa", "cushion"],
        },
        {
            "kind": "proxy",
            "rank": 2,
            "grounding_query": {
                "raw_query": "the pillow on the bed",
                "root": {
                    "categories": ["pillow", "throw_pillow"],
                    "attributes": [],
                    "spatial_constraints": [
                        {
                            "relation": "on",
                            "anchors": [
                                {
                                    "categories": ["armchair"],
                                    "attributes": [],
                                    "spatial_constraints": [],
                                    "select_constraint": None,
                                }
                            ],
                        }
                    ],
                    "select_constraint": None,
                },
                "expect_unique": True,
            },
            "lexical_hints": ["pillow", "armchair"],
        },
        {
            "kind": "context",
            "rank": 3,
            "grounding_query": {
                "raw_query": "pillows in living room",
                "root": {
                    "categories": ["pillow", "throw_pillow"],
                    "attributes": [],
                    "spatial_constraints": [],
                    "select_constraint": None,
                },
                "expect_unique": False,
            },
            "lexical_hints": ["pillow", "living room"],
        },
    ]


@pytest.fixture
def sample_bundle(sample_hypotheses):
    """Create a sample Stage2EvidenceBundle with hypotheses."""
    return Stage2EvidenceBundle(
        scene_id="room0",
        stage1_query="the pillow on the sofa",
        keyframes=[
            KeyframeEvidence(
                keyframe_idx=0,
                image_path="/test/frame_000.jpg",
                view_id=0,
                frame_id=0,
                score=0.85,
                note="direct hypothesis match",
            ),
            KeyframeEvidence(
                keyframe_idx=1,
                image_path="/test/frame_001.jpg",
                view_id=1,
                frame_id=10,
                score=0.72,
                note="secondary match",
            ),
        ],
        bev_image_path="/test/bev.png",
        scene_summary="Living room with sofa, pillows, and armchair",
        object_context={
            "pillow": "Decorative pillow on sofa",
            "sofa": "Gray 3-seater sofa",
            "armchair": "Single armchair near window",
        },
        hypothesis=Stage1HypothesisSummary(
            status="success",
            hypothesis_kind="direct",
            hypothesis_rank=1,
            parse_mode="multi",
            raw_query="the pillow on the sofa",
            target_categories=["pillow", "throw_pillow"],
            anchor_categories=["sofa"],
            metadata={"version": "v3"},
        ),
        extra_metadata={
            "status": "success",
            "hypothesis_output": {
                "format_version": "hypothesis_output_v1",
                "parse_mode": "multi",
                "hypotheses": sample_hypotheses,
            },
            "selected_hypothesis_kind": "direct",
            "selected_hypothesis_rank": 1,
        },
    )


@pytest.fixture
def single_hypothesis_bundle():
    """Create a bundle with only direct hypothesis (single mode)."""
    return Stage2EvidenceBundle(
        scene_id="room0",
        stage1_query="the sofa",
        keyframes=[
            KeyframeEvidence(
                keyframe_idx=0,
                image_path="/test/frame_000.jpg",
                view_id=0,
                frame_id=0,
                score=0.95,
            )
        ],
        hypothesis=Stage1HypothesisSummary(
            status="success",
            hypothesis_kind="direct",
            hypothesis_rank=1,
            parse_mode="single",
            raw_query="the sofa",
            target_categories=["sofa"],
            anchor_categories=[],
        ),
        extra_metadata={
            "status": "success",
            "hypothesis_output": {
                "format_version": "hypothesis_output_v1",
                "parse_mode": "single",
                "hypotheses": [
                    {
                        "kind": "direct",
                        "rank": 1,
                        "grounding_query": {
                            "raw_query": "the sofa",
                            "root": {
                                "categories": ["sofa"],
                                "attributes": [],
                                "spatial_constraints": [],
                                "select_constraint": None,
                            },
                            "expect_unique": True,
                        },
                        "lexical_hints": ["sofa", "couch"],
                    }
                ],
            },
            "selected_hypothesis_kind": "direct",
            "selected_hypothesis_rank": 1,
        },
    )


class TestHypothesisHistoryEntry:
    """Tests for HypothesisHistoryEntry dataclass."""

    def test_create_entry(self):
        """Test creating a history entry."""
        entry = HypothesisHistoryEntry(
            timestamp=1234567890.0,
            hypothesis_kind="direct",
            hypothesis_rank=1,
            action=HypothesisAction.SWITCH,
            reason="Testing switch",
            success=True,
            metadata={"test": "value"},
        )
        assert entry.hypothesis_kind == "direct"
        assert entry.action == HypothesisAction.SWITCH
        assert entry.success is True

    def test_to_dict(self):
        """Test converting entry to dictionary."""
        entry = HypothesisHistoryEntry(
            timestamp=1234567890.0,
            hypothesis_kind="proxy",
            hypothesis_rank=2,
            action=HypothesisAction.EXPAND,
            reason="Need more options",
            success=False,
            metadata={"error": "callback not configured"},
        )
        d = entry.to_dict()
        assert d["hypothesis_kind"] == "proxy"
        assert d["action"] == "expand"
        assert d["success"] is False
        assert d["metadata"]["error"] == "callback not configured"


class TestHypothesisRepairConfig:
    """Tests for HypothesisRepairConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HypothesisRepairConfig()
        assert config.max_history_entries == 20
        assert config.allow_regeneration is True
        assert config.default_fallback_order == ["direct", "proxy", "context"]

    def test_custom_config(self):
        """Test custom configuration."""
        config = HypothesisRepairConfig(
            max_history_entries=10,
            allow_regeneration=False,
            default_fallback_order=["proxy", "context"],
        )
        assert config.max_history_entries == 10
        assert config.allow_regeneration is False
        assert config.default_fallback_order == ["proxy", "context"]


class TestHypothesisRepairBackend:
    """Tests for HypothesisRepairBackend class."""

    def test_init_default(self):
        """Test default initialization."""
        backend = HypothesisRepairBackend()
        assert backend.config.max_history_entries == 20
        assert backend.stage1_reparse_callback is None
        assert len(backend.history) == 0

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = HypothesisRepairConfig(max_history_entries=5)
        backend = HypothesisRepairBackend(config=config)
        assert backend.config.max_history_entries == 5

    def test_history_management(self):
        """Test history tracking and clearing."""
        backend = HypothesisRepairBackend()
        backend._add_history_entry(
            hypothesis_kind="direct",
            hypothesis_rank=1,
            action=HypothesisAction.SWITCH,
            reason="test",
            success=True,
        )
        assert len(backend.history) == 1
        backend.clear_history()
        assert len(backend.history) == 0

    def test_history_size_limit(self):
        """Test history size limiting."""
        config = HypothesisRepairConfig(max_history_entries=3)
        backend = HypothesisRepairBackend(config=config)

        for i in range(5):
            backend._add_history_entry(
                hypothesis_kind="direct",
                hypothesis_rank=i,
                action=HypothesisAction.INSPECT,
                reason=f"test {i}",
                success=True,
            )

        assert len(backend.history) == 3
        # Should keep the most recent entries
        assert backend.history[0].hypothesis_rank == 2
        assert backend.history[-1].hypothesis_rank == 4


class TestInspectHypothesisState:
    """Tests for inspect_hypothesis_state method."""

    def test_inspect_multi_hypothesis(self, sample_bundle):
        """Test inspecting bundle with multiple hypotheses."""
        backend = HypothesisRepairBackend()
        state = backend.inspect_hypothesis_state(sample_bundle)

        assert state["current_hypothesis"] is not None
        assert state["current_hypothesis"]["hypothesis_kind"] == "direct"
        assert len(state["available_hypotheses"]) == 3
        assert state["parse_mode"] == "multi"
        assert len(state["recommendations"]) > 0

    def test_inspect_single_hypothesis(self, single_hypothesis_bundle):
        """Test inspecting bundle with single hypothesis."""
        backend = HypothesisRepairBackend()
        state = backend.inspect_hypothesis_state(single_hypothesis_bundle)

        assert state["current_hypothesis"]["hypothesis_kind"] == "direct"
        assert len(state["available_hypotheses"]) == 1
        assert state["parse_mode"] == "single"

    def test_inspect_shows_history(self, sample_bundle):
        """Test that inspect shows recent history."""
        backend = HypothesisRepairBackend()
        backend._add_history_entry(
            hypothesis_kind="direct",
            hypothesis_rank=1,
            action=HypothesisAction.INSPECT,
            reason="initial inspection",
            success=True,
        )

        state = backend.inspect_hypothesis_state(sample_bundle)
        assert state["total_history_entries"] == 1
        assert len(state["recent_history"]) == 1


class TestSwitchHypothesis:
    """Tests for switch_hypothesis method."""

    def test_switch_to_proxy(self, sample_bundle):
        """Test switching from direct to proxy hypothesis."""
        backend = HypothesisRepairBackend()
        success, updated_bundle, message = backend.switch_hypothesis(
            sample_bundle, "proxy", reason="direct match insufficient"
        )

        assert success is True
        assert "proxy" in message.lower()
        assert updated_bundle.hypothesis.hypothesis_kind == "proxy"
        assert updated_bundle.hypothesis.hypothesis_rank == 2
        assert len(backend.history) == 1

    def test_switch_to_context(self, sample_bundle):
        """Test switching from direct to context hypothesis."""
        backend = HypothesisRepairBackend()
        success, updated_bundle, message = backend.switch_hypothesis(
            sample_bundle, "context", reason="need broader search"
        )

        assert success is True
        assert updated_bundle.hypothesis.hypothesis_kind == "context"
        assert updated_bundle.hypothesis.hypothesis_rank == 3

    def test_switch_to_unavailable(self, single_hypothesis_bundle):
        """Test switching to unavailable hypothesis kind."""
        backend = HypothesisRepairBackend()
        success, updated_bundle, message = backend.switch_hypothesis(
            single_hypothesis_bundle, "proxy", reason="trying fallback"
        )

        assert success is False
        assert "not available" in message.lower()
        assert updated_bundle is single_hypothesis_bundle  # Unchanged
        assert len(backend.history) == 1
        assert backend.history[0].success is False

    def test_switch_preserves_metadata(self, sample_bundle):
        """Test that switch preserves important metadata."""
        backend = HypothesisRepairBackend()
        success, updated_bundle, message = backend.switch_hypothesis(
            sample_bundle, "proxy"
        )

        assert success is True
        assert updated_bundle.scene_id == sample_bundle.scene_id
        assert updated_bundle.stage1_query == sample_bundle.stage1_query
        assert len(updated_bundle.keyframes) == len(sample_bundle.keyframes)
        # Metadata should reflect the switch
        assert updated_bundle.hypothesis.metadata.get("switched_from") == "direct"


class TestGetNextFallbackHypothesis:
    """Tests for get_next_fallback_hypothesis method."""

    def test_fallback_from_direct(self, sample_bundle):
        """Test getting next fallback from direct hypothesis."""
        backend = HypothesisRepairBackend()
        next_kind = backend.get_next_fallback_hypothesis(sample_bundle)
        assert next_kind == "proxy"

    def test_fallback_from_proxy(self, sample_bundle):
        """Test getting next fallback from proxy hypothesis."""
        backend = HypothesisRepairBackend()
        # Switch to proxy first
        _, updated_bundle, _ = backend.switch_hypothesis(sample_bundle, "proxy")
        next_kind = backend.get_next_fallback_hypothesis(updated_bundle)
        assert next_kind == "context"

    def test_fallback_exhausted(self, sample_bundle):
        """Test fallback when all hypotheses exhausted."""
        backend = HypothesisRepairBackend()
        # Switch to context
        _, updated_bundle, _ = backend.switch_hypothesis(sample_bundle, "context")
        next_kind = backend.get_next_fallback_hypothesis(updated_bundle)
        assert next_kind is None

    def test_fallback_single_mode(self, single_hypothesis_bundle):
        """Test fallback with single hypothesis (no fallbacks available)."""
        backend = HypothesisRepairBackend()
        next_kind = backend.get_next_fallback_hypothesis(single_hypothesis_bundle)
        assert next_kind is None


class TestRequestExpansion:
    """Tests for request_expansion method."""

    def test_expansion_without_callback(self, sample_bundle):
        """Test expansion request without Stage-1 callback."""
        backend = HypothesisRepairBackend()
        success, updated_bundle, message = backend.request_expansion(
            sample_bundle, reason="need more options"
        )

        assert success is False
        assert "not configured" in message.lower()
        assert updated_bundle is sample_bundle  # Unchanged

    def test_expansion_with_callback(self, sample_bundle):
        """Test expansion with Stage-1 callback."""

        def mock_callback(bundle, constraints):
            # Simulate adding new hypotheses
            updated = bundle.model_copy(deep=True)
            return updated

        backend = HypothesisRepairBackend(stage1_reparse_callback=mock_callback)
        success, updated_bundle, message = backend.request_expansion(
            sample_bundle, expansion_hints=["bedroom"], reason="try bedroom context"
        )

        assert success is True
        assert len(backend.history) == 1
        assert backend.history[0].action == HypothesisAction.EXPAND

    def test_expansion_callback_failure(self, sample_bundle):
        """Test expansion when callback raises exception."""

        def failing_callback(bundle, constraints):
            raise ValueError("Stage-1 parsing failed")

        backend = HypothesisRepairBackend(stage1_reparse_callback=failing_callback)
        success, updated_bundle, message = backend.request_expansion(sample_bundle)

        assert success is False
        assert "failed" in message.lower()


class TestHandleToolRequest:
    """Tests for handle_tool_request method."""

    def test_inspect_request(self, sample_bundle):
        """Test handling inspect request."""
        backend = HypothesisRepairBackend()
        result = backend.handle_tool_request(
            sample_bundle,
            {"request_text": "show current hypothesis status", "preferred_kind": ""},
        )

        assert "current_hypothesis" in result.response_text
        assert result.updated_bundle is None

    def test_switch_request_with_preferred_kind(self, sample_bundle):
        """Test handling switch request with explicit kind."""
        backend = HypothesisRepairBackend()
        result = backend.handle_tool_request(
            sample_bundle,
            {"request_text": "change hypothesis", "preferred_kind": "proxy"},
        )

        assert result.updated_bundle is not None
        assert result.updated_bundle.hypothesis.hypothesis_kind == "proxy"

    def test_switch_request_from_text(self, sample_bundle):
        """Test handling switch request parsed from text."""
        backend = HypothesisRepairBackend()
        result = backend.handle_tool_request(
            sample_bundle,
            {
                "request_text": "try the proxy hypothesis for alternative matching",
                "preferred_kind": "",
            },
        )

        assert result.updated_bundle is not None
        assert result.updated_bundle.hypothesis.hypothesis_kind == "proxy"

    def test_automatic_fallback(self, sample_bundle):
        """Test automatic fallback when no specific kind requested."""
        backend = HypothesisRepairBackend()
        result = backend.handle_tool_request(
            sample_bundle,
            {"request_text": "switch to fallback hypothesis", "preferred_kind": ""},
        )

        assert result.updated_bundle is not None
        # Should switch to proxy (next in fallback order)
        assert result.updated_bundle.hypothesis.hypothesis_kind == "proxy"

    def test_expand_request(self, sample_bundle):
        """Test handling expand request."""
        backend = HypothesisRepairBackend()
        result = backend.handle_tool_request(
            sample_bundle,
            {
                "request_text": "expand hypotheses with more alternatives",
                "preferred_kind": "",
            },
        )

        # Without callback, should fail gracefully
        assert "not configured" in result.response_text.lower()

    def test_semantic_switch_hints(self, sample_bundle):
        """Test semantic hints for switching ('exact', 'broader', etc.)."""
        backend = HypothesisRepairBackend()

        # Test "broader" -> context
        result = backend.handle_tool_request(
            sample_bundle,
            {"request_text": "try a broader search approach", "preferred_kind": ""},
        )
        assert result.updated_bundle.hypothesis.hypothesis_kind == "context"

        # Reset and test "similar" -> proxy
        backend.clear_history()
        result = backend.handle_tool_request(
            sample_bundle,
            {
                "request_text": "try searching for similar objects",
                "preferred_kind": "",
            },
        )
        assert result.updated_bundle.hypothesis.hypothesis_kind == "proxy"


class TestParseAction:
    """Tests for _parse_action method."""

    def test_parse_inspect(self):
        """Test parsing inspect action."""
        backend = HypothesisRepairBackend()
        assert (
            backend._parse_action("show me the status", "") == HypothesisAction.INSPECT
        )
        assert (
            backend._parse_action("what is the current hypothesis?", "")
            == HypothesisAction.INSPECT
        )

    def test_parse_switch(self):
        """Test parsing switch action."""
        backend = HypothesisRepairBackend()
        assert backend._parse_action("switch to proxy", "") == HypothesisAction.SWITCH
        assert (
            backend._parse_action("try the context hypothesis", "")
            == HypothesisAction.SWITCH
        )
        assert backend._parse_action("use direct", "") == HypothesisAction.SWITCH
        # With preferred_kind
        assert backend._parse_action("", "proxy") == HypothesisAction.SWITCH

    def test_parse_expand(self):
        """Test parsing expand action."""
        backend = HypothesisRepairBackend()
        assert backend._parse_action("expand hypotheses", "") == HypothesisAction.EXPAND
        assert (
            backend._parse_action("need more hypotheses", "") == HypothesisAction.EXPAND
        )
        assert (
            backend._parse_action("get additional alternatives", "")
            == HypothesisAction.EXPAND
        )

    def test_parse_regenerate(self):
        """Test parsing regenerate action."""
        backend = HypothesisRepairBackend()
        assert (
            backend._parse_action("regenerate the parsing", "")
            == HypothesisAction.REGENERATE
        )
        assert (
            backend._parse_action("re-parse the query", "")
            == HypothesisAction.REGENERATE
        )


class TestExtractExpansionHints:
    """Tests for _extract_expansion_hints method."""

    def test_extract_quoted_hints(self):
        """Test extracting quoted hints."""
        backend = HypothesisRepairBackend()
        hints = backend._extract_expansion_hints('try "bedroom" and "bed"')
        assert "bedroom" in hints
        assert "bed" in hints

    def test_extract_single_quoted_hints(self):
        """Test extracting single-quoted hints."""
        backend = HypothesisRepairBackend()
        hints = backend._extract_expansion_hints("try 'living room' context")
        assert "living room" in hints

    def test_deduplicate_hints(self):
        """Test that hints are deduplicated."""
        backend = HypothesisRepairBackend()
        hints = backend._extract_expansion_hints('"sofa" and also "Sofa"')
        assert len(hints) == 1


class TestCreateHypothesisRepairCallback:
    """Tests for the convenience callback creation."""

    def test_create_callback(self, sample_bundle):
        """Test creating and using the callback."""
        callback = create_hypothesis_repair_callback()
        result = callback(
            sample_bundle,
            {"request_text": "inspect", "preferred_kind": ""},
        )
        assert "current_hypothesis" in result.response_text

    def test_callback_state_isolation(self, sample_bundle):
        """Test that different callbacks have isolated state."""
        callback1 = create_hypothesis_repair_callback()
        callback2 = create_hypothesis_repair_callback()

        # Use callback1
        callback1(
            sample_bundle,
            {"request_text": "switch to proxy", "preferred_kind": "proxy"},
        )

        # callback2 should have fresh state (no history)
        result = callback2(
            sample_bundle,
            {"request_text": "inspect", "preferred_kind": ""},
        )
        import json

        state = json.loads(result.response_text)
        assert state["total_history_entries"] == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_bundle(self):
        """Test handling bundle with no hypotheses."""
        backend = HypothesisRepairBackend()
        bundle = Stage2EvidenceBundle(
            scene_id="test",
            stage1_query="test query",
            keyframes=[],
            extra_metadata={},
        )

        state = backend.inspect_hypothesis_state(bundle)
        assert state["current_hypothesis"] is None
        assert len(state["available_hypotheses"]) == 0

    def test_bundle_without_hypothesis_field(self):
        """Test handling bundle without hypothesis summary."""
        backend = HypothesisRepairBackend()
        bundle = Stage2EvidenceBundle(
            scene_id="test",
            stage1_query="test query",
            keyframes=[],
            hypothesis=None,
            extra_metadata={
                "hypothesis_output": {
                    "hypotheses": [
                        {
                            "kind": "direct",
                            "rank": 1,
                            "grounding_query": {
                                "raw_query": "test",
                                "root": {"categories": ["test"]},
                            },
                        }
                    ]
                }
            },
        )

        # Should still be able to get available hypotheses
        state = backend.inspect_hypothesis_state(bundle)
        assert len(state["available_hypotheses"]) == 1

    def test_switch_preserves_keyframes(self, sample_bundle):
        """Test that keyframes are preserved during switch."""
        backend = HypothesisRepairBackend()
        original_keyframes = len(sample_bundle.keyframes)

        success, updated_bundle, _ = backend.switch_hypothesis(sample_bundle, "proxy")

        assert success is True
        assert len(updated_bundle.keyframes) == original_keyframes
        assert (
            updated_bundle.keyframes[0].image_path
            == sample_bundle.keyframes[0].image_path
        )

    def test_regeneration_disabled(self, sample_bundle):
        """Test that regeneration can be disabled."""
        config = HypothesisRepairConfig(allow_regeneration=False)
        backend = HypothesisRepairBackend(config=config)

        result = backend.handle_tool_request(
            sample_bundle,
            {"request_text": "regenerate hypotheses", "preferred_kind": ""},
        )

        assert "not enabled" in result.response_text.lower()
