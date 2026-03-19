"""Backend implementation for the switch_or_expand_hypothesis Stage-2 tool.

This module provides the hypothesis repair capability for the Stage-2 VLM agent.
It supports:
1. Switching between direct/proxy/context hypotheses
2. Requesting alternative hypotheses from Stage 1
3. Tracking hypothesis history for analysis

Design rationale (Academic alignment):
- Supports "symbolic-to-visual repair" by enabling the agent to validate and
  correct Stage-1 hypotheses using visual evidence
- When a direct hypothesis fails visual verification, the agent can switch to
  proxy or context hypotheses as fallback strategies
- Enables "adaptive evidence acquisition" by letting the agent request
  alternative interpretations of the query
- Works across all task types (QA, grounding, nav, manipulation)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from ..models import (
    Stage1HypothesisSummary,
    Stage2EvidenceBundle,
    Stage2ToolResult,
)


class HypothesisAction(str, Enum):
    """Actions that can be performed on hypotheses."""

    SWITCH = "switch"  # Switch to a different hypothesis kind
    EXPAND = "expand"  # Request additional hypotheses from Stage 1
    REGENERATE = "regenerate"  # Request complete re-parsing with modified constraints
    INSPECT = "inspect"  # Just inspect current hypothesis state


@dataclass
class HypothesisHistoryEntry:
    """Record of a single hypothesis state in the repair history.

    This enables tracking how hypotheses evolve during Stage-2 reasoning,
    supporting academic analysis of symbolic-to-visual repair patterns.
    """

    timestamp: float
    hypothesis_kind: str
    hypothesis_rank: Optional[int]
    action: HypothesisAction
    reason: str
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "hypothesis_kind": self.hypothesis_kind,
            "hypothesis_rank": self.hypothesis_rank,
            "action": self.action.value,
            "reason": self.reason,
            "success": self.success,
            "metadata": self.metadata,
        }


@dataclass
class HypothesisRepairConfig:
    """Configuration for the hypothesis repair backend."""

    max_history_entries: int = 20  # Maximum history entries to retain
    allow_regeneration: bool = True  # Allow full re-parsing requests
    default_fallback_order: List[str] = field(
        default_factory=lambda: ["direct", "proxy", "context"]
    )


class HypothesisRepairBackend:
    """Backend for hypothesis switching, expansion, and tracking.

    This backend implements the real hypothesis repair capability:
    1. Inspects current Stage-1 hypothesis metadata
    2. Switches between available hypotheses (direct/proxy/context)
    3. Requests alternative hypotheses from Stage 1 when needed
    4. Tracks hypothesis history for academic analysis

    Usage:
        backend = HypothesisRepairBackend()
        callback = backend.create_callback()
        agent = Stage2DeepResearchAgent(hypothesis_callback=callback)

    Academic alignment:
        - Supports "symbolic-to-visual repair": Stage-2 can validate and correct
          Stage-1 hypotheses by switching to alternatives when visual evidence
          contradicts the current hypothesis
        - Enables analysis of repair patterns: which hypothesis types succeed,
          when proxy replaces direct, etc.
    """

    def __init__(
        self,
        config: Optional[HypothesisRepairConfig] = None,
        stage1_reparse_callback: Optional[
            Callable[[Stage2EvidenceBundle, Dict[str, Any]], Stage2EvidenceBundle]
        ] = None,
    ) -> None:
        """Initialize hypothesis repair backend.

        Args:
            config: Backend configuration
            stage1_reparse_callback: Optional callback to request Stage-1 re-parsing.
                Signature: (bundle, constraints) -> updated_bundle
                If not provided, expansion/regeneration will return guidance only.
        """
        self.config = config or HypothesisRepairConfig()
        self.stage1_reparse_callback = stage1_reparse_callback
        self._history: List[HypothesisHistoryEntry] = []

    @property
    def history(self) -> List[HypothesisHistoryEntry]:
        """Get the hypothesis repair history."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear the hypothesis repair history."""
        self._history.clear()

    def _add_history_entry(
        self,
        hypothesis_kind: str,
        hypothesis_rank: Optional[int],
        action: HypothesisAction,
        reason: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HypothesisHistoryEntry:
        """Add an entry to the history and maintain size limit."""
        entry = HypothesisHistoryEntry(
            timestamp=time.time(),
            hypothesis_kind=hypothesis_kind,
            hypothesis_rank=hypothesis_rank,
            action=action,
            reason=reason,
            success=success,
            metadata=metadata or {},
        )
        self._history.append(entry)

        # Trim history if exceeds limit
        if len(self._history) > self.config.max_history_entries:
            self._history = self._history[-self.config.max_history_entries :]

        return entry

    def _get_available_hypotheses(
        self, bundle: Stage2EvidenceBundle
    ) -> List[Dict[str, Any]]:
        """Extract available hypotheses from the bundle's extra_metadata."""
        hypothesis_output = bundle.extra_metadata.get("hypothesis_output", {})
        hypotheses = hypothesis_output.get("hypotheses", [])
        return hypotheses

    def _get_hypothesis_by_kind(
        self,
        hypotheses: List[Dict[str, Any]],
        kind: str,
    ) -> Optional[Dict[str, Any]]:
        """Find a hypothesis by its kind."""
        for h in hypotheses:
            if h.get("kind") == kind:
                return h
        return None

    def _get_hypothesis_by_rank(
        self,
        hypotheses: List[Dict[str, Any]],
        rank: int,
    ) -> Optional[Dict[str, Any]]:
        """Find a hypothesis by its rank."""
        for h in hypotheses:
            if h.get("rank") == rank:
                return h
        return None

    def inspect_hypothesis_state(
        self,
        bundle: Stage2EvidenceBundle,
    ) -> Dict[str, Any]:
        """Inspect the current hypothesis state without making changes.

        Returns a detailed view of:
        - Current selected hypothesis
        - All available hypotheses
        - Hypothesis repair history
        - Recommendations for next action
        """
        hypotheses = self._get_available_hypotheses(bundle)
        current = bundle.hypothesis

        # Build hypothesis summary list
        available = []
        for h in hypotheses:
            grounding_query = h.get("grounding_query", {})
            root = grounding_query.get("root", {})
            available.append(
                {
                    "kind": h.get("kind"),
                    "rank": h.get("rank"),
                    "target_categories": root.get("categories", []),
                    "has_spatial_constraints": bool(root.get("spatial_constraints")),
                    "has_select_constraint": root.get("select_constraint") is not None,
                    "lexical_hints": h.get("lexical_hints", []),
                }
            )

        # Build recommendations
        recommendations = []
        current_kind = current.hypothesis_kind if current else None

        if current_kind == "direct":
            if self._get_hypothesis_by_kind(hypotheses, "proxy"):
                recommendations.append(
                    "If direct hypothesis constraints are too strict, try 'proxy' "
                    "which uses similar objects as anchors."
                )
            if self._get_hypothesis_by_kind(hypotheses, "context"):
                recommendations.append(
                    "If target object is not found, try 'context' "
                    "which searches for contextual scene elements."
                )
        elif current_kind == "proxy":
            recommendations.append(
                "If proxy hypothesis is insufficient, consider 'context' "
                "for broader scene exploration."
            )
            if self._get_hypothesis_by_kind(hypotheses, "direct"):
                recommendations.append(
                    "If you want stricter matching, switch back to 'direct'."
                )
        elif current_kind == "context":
            recommendations.append(
                "Context mode provides broadest coverage. Consider requesting "
                "additional keyframes if evidence is still insufficient."
            )

        # Recent history summary
        recent_history = [e.to_dict() for e in self._history[-5:]]

        return {
            "current_hypothesis": current.model_dump() if current else None,
            "available_hypotheses": available,
            "parse_mode": bundle.extra_metadata.get("hypothesis_output", {}).get(
                "parse_mode"
            ),
            "recommendations": recommendations,
            "recent_history": recent_history,
            "total_history_entries": len(self._history),
        }

    def switch_hypothesis(
        self,
        bundle: Stage2EvidenceBundle,
        target_kind: str,
        reason: str = "",
    ) -> Tuple[bool, Stage2EvidenceBundle, str]:
        """Switch to a different hypothesis kind.

        Args:
            bundle: Current evidence bundle
            target_kind: Target hypothesis kind ('direct', 'proxy', 'context')
            reason: Reason for switching (for history tracking)

        Returns:
            Tuple of (success, updated_bundle, message)
        """
        hypotheses = self._get_available_hypotheses(bundle)
        target = self._get_hypothesis_by_kind(hypotheses, target_kind)

        if target is None:
            self._add_history_entry(
                hypothesis_kind=target_kind,
                hypothesis_rank=None,
                action=HypothesisAction.SWITCH,
                reason=reason,
                success=False,
                metadata={"error": f"No hypothesis of kind '{target_kind}' available"},
            )
            available_kinds = [h.get("kind") for h in hypotheses]
            return (
                False,
                bundle,
                f"Cannot switch to '{target_kind}': not available. "
                f"Available kinds: {available_kinds}",
            )

        # Build updated hypothesis summary
        grounding_query = target.get("grounding_query", {})
        root = grounding_query.get("root", {})
        target_categories = list(root.get("categories", []))
        anchor_categories: List[str] = []
        for constraint in root.get("spatial_constraints", []):
            for anchor in constraint.get("anchors", []):
                anchor_categories.extend(anchor.get("categories", []))

        new_hypothesis = Stage1HypothesisSummary(
            status=bundle.extra_metadata.get("status", "switched"),
            hypothesis_kind=target.get("kind", ""),
            hypothesis_rank=target.get("rank"),
            parse_mode=bundle.extra_metadata.get("hypothesis_output", {}).get(
                "parse_mode", ""
            ),
            raw_query=grounding_query.get("raw_query", bundle.stage1_query),
            target_categories=target_categories,
            anchor_categories=anchor_categories,
            metadata={
                "switched_from": (
                    bundle.hypothesis.hypothesis_kind if bundle.hypothesis else None
                ),
                "switch_reason": reason,
            },
        )

        # Create updated bundle
        updated_bundle = bundle.model_copy(deep=True)
        updated_bundle.hypothesis = new_hypothesis

        # Update extra_metadata to reflect selected hypothesis
        if "selected_hypothesis_kind" in updated_bundle.extra_metadata:
            updated_bundle.extra_metadata["selected_hypothesis_kind"] = target.get(
                "kind"
            )
        if "selected_hypothesis_rank" in updated_bundle.extra_metadata:
            updated_bundle.extra_metadata["selected_hypothesis_rank"] = target.get(
                "rank"
            )

        self._add_history_entry(
            hypothesis_kind=target.get("kind", ""),
            hypothesis_rank=target.get("rank"),
            action=HypothesisAction.SWITCH,
            reason=reason,
            success=True,
            metadata={
                "target_categories": target_categories,
                "anchor_categories": anchor_categories,
            },
        )

        return (
            True,
            updated_bundle,
            f"Switched to '{target_kind}' hypothesis (rank {target.get('rank')}). "
            f"Target categories: {target_categories}",
        )

    def request_expansion(
        self,
        bundle: Stage2EvidenceBundle,
        expansion_hints: Optional[List[str]] = None,
        reason: str = "",
    ) -> Tuple[bool, Stage2EvidenceBundle, str]:
        """Request additional hypotheses from Stage 1.

        If a stage1_reparse_callback is configured, this will request new
        hypotheses. Otherwise, it provides guidance on what to try.

        Args:
            bundle: Current evidence bundle
            expansion_hints: Optional hints for generating new hypotheses
            reason: Reason for expansion request

        Returns:
            Tuple of (success, updated_bundle, message)
        """
        if self.stage1_reparse_callback is None:
            self._add_history_entry(
                hypothesis_kind=(
                    bundle.hypothesis.hypothesis_kind
                    if bundle.hypothesis
                    else "unknown"
                ),
                hypothesis_rank=(
                    bundle.hypothesis.hypothesis_rank if bundle.hypothesis else None
                ),
                action=HypothesisAction.EXPAND,
                reason=reason,
                success=False,
                metadata={"error": "No Stage-1 callback configured"},
            )

            # Provide guidance instead
            hypotheses = self._get_available_hypotheses(bundle)
            current_kinds = {h.get("kind") for h in hypotheses}
            missing_kinds = {"direct", "proxy", "context"} - current_kinds

            guidance = [
                "Stage-1 re-parsing callback is not configured.",
                f"Current available hypothesis types: {sorted(current_kinds)}",
            ]
            if missing_kinds:
                guidance.append(
                    f"Missing hypothesis types that could be requested: {sorted(missing_kinds)}"
                )
            guidance.append(
                "Consider using request_more_views to gather alternative visual evidence."
            )

            return (False, bundle, "\n".join(guidance))

        # Call the Stage-1 callback
        constraints = {
            "expansion_hints": expansion_hints or [],
            "current_hypothesis_kind": (
                bundle.hypothesis.hypothesis_kind if bundle.hypothesis else None
            ),
            "reason": reason,
        }

        try:
            updated_bundle = self.stage1_reparse_callback(bundle, constraints)

            self._add_history_entry(
                hypothesis_kind="expanded",
                hypothesis_rank=None,
                action=HypothesisAction.EXPAND,
                reason=reason,
                success=True,
                metadata={
                    "expansion_hints": expansion_hints,
                    "new_hypothesis_count": len(
                        updated_bundle.extra_metadata.get("hypothesis_output", {}).get(
                            "hypotheses", []
                        )
                    ),
                },
            )

            new_hypotheses = self._get_available_hypotheses(updated_bundle)
            kinds = [h.get("kind") for h in new_hypotheses]
            return (
                True,
                updated_bundle,
                f"Expanded hypotheses. Now available: {kinds}",
            )
        except Exception as e:
            self._add_history_entry(
                hypothesis_kind="expanded",
                hypothesis_rank=None,
                action=HypothesisAction.EXPAND,
                reason=reason,
                success=False,
                metadata={"error": str(e)},
            )
            return (False, bundle, f"Expansion failed: {e}")

    def get_next_fallback_hypothesis(
        self,
        bundle: Stage2EvidenceBundle,
    ) -> Optional[str]:
        """Get the next hypothesis kind in the fallback order.

        This helps the agent systematically try hypotheses in order:
        direct -> proxy -> context

        Returns:
            Next hypothesis kind to try, or None if all exhausted
        """
        hypotheses = self._get_available_hypotheses(bundle)
        available_kinds = {h.get("kind") for h in hypotheses}
        current_kind = bundle.hypothesis.hypothesis_kind if bundle.hypothesis else None

        # Find current position in fallback order
        fallback_order = self.config.default_fallback_order
        if current_kind in fallback_order:
            current_idx = fallback_order.index(current_kind)
            # Try remaining kinds in order
            for kind in fallback_order[current_idx + 1 :]:
                if kind in available_kinds:
                    return kind
        else:
            # Current kind not in fallback order, try from beginning
            for kind in fallback_order:
                if kind in available_kinds and kind != current_kind:
                    return kind

        return None

    def handle_tool_request(
        self,
        bundle: Stage2EvidenceBundle,
        request_dict: Dict[str, Any],
    ) -> Stage2ToolResult:
        """Handle a tool request from the Stage-2 agent.

        This is the main entry point called by the agent's switch_or_expand_hypothesis tool.

        Args:
            bundle: Current evidence bundle
            request_dict: Raw request dictionary from the agent tool call
                - request_text: Free-form description of what the agent wants
                - preferred_kind: Optional specific hypothesis kind to switch to

        Returns:
            Stage2ToolResult with response text and optional updated bundle
        """
        request_text = request_dict.get("request_text", "")
        preferred_kind = request_dict.get("preferred_kind", "")

        # Parse the request to determine action
        action = self._parse_action(request_text, preferred_kind)

        if action == HypothesisAction.INSPECT:
            state = self.inspect_hypothesis_state(bundle)
            return Stage2ToolResult(
                response_text=json.dumps(state, indent=2, ensure_ascii=False),
            )

        if action == HypothesisAction.SWITCH:
            target_kind = self._determine_switch_target(
                bundle, request_text, preferred_kind
            )
            if target_kind is None:
                # Try automatic fallback
                target_kind = self.get_next_fallback_hypothesis(bundle)
                if target_kind is None:
                    return Stage2ToolResult(
                        response_text="No alternative hypothesis available to switch to. "
                        "All hypothesis types have been exhausted."
                    )

            success, updated_bundle, message = self.switch_hypothesis(
                bundle, target_kind, reason=request_text
            )

            return Stage2ToolResult(
                response_text=message,
                updated_bundle=updated_bundle if success else None,
            )

        if action == HypothesisAction.EXPAND:
            hints = self._extract_expansion_hints(request_text)
            success, updated_bundle, message = self.request_expansion(
                bundle, expansion_hints=hints, reason=request_text
            )
            return Stage2ToolResult(
                response_text=message,
                updated_bundle=updated_bundle if success else None,
            )

        if action == HypothesisAction.REGENERATE:
            if not self.config.allow_regeneration:
                return Stage2ToolResult(
                    response_text="Hypothesis regeneration is not enabled in this configuration. "
                    "Try switching between available hypotheses or requesting more views."
                )
            # Regeneration is handled similarly to expansion
            hints = self._extract_expansion_hints(request_text)
            success, updated_bundle, message = self.request_expansion(
                bundle, expansion_hints=hints, reason=f"regenerate: {request_text}"
            )
            return Stage2ToolResult(
                response_text=message,
                updated_bundle=updated_bundle if success else None,
            )

        # Fallback: inspect
        state = self.inspect_hypothesis_state(bundle)
        return Stage2ToolResult(
            response_text="Could not determine requested action. "
            "Current hypothesis state:\n"
            + json.dumps(state, indent=2, ensure_ascii=False),
        )

    def _parse_action(
        self,
        request_text: str,
        preferred_kind: str,
    ) -> HypothesisAction:
        """Parse the agent's request to determine the action."""
        text_lower = request_text.lower()

        # Check for specific keywords
        if any(kw in text_lower for kw in ["inspect", "status", "show", "what is"]):
            return HypothesisAction.INSPECT

        if any(
            kw in text_lower
            for kw in ["regenerate", "reparse", "re-parse", "new parsing"]
        ):
            return HypothesisAction.REGENERATE

        if any(kw in text_lower for kw in ["expand", "more hypotheses", "additional"]):
            return HypothesisAction.EXPAND

        if (
            any(
                kw in text_lower
                for kw in ["switch", "try", "use", "change to", "fallback"]
            )
            or preferred_kind
        ):
            return HypothesisAction.SWITCH

        # Default to switch if preferred_kind is specified
        if preferred_kind:
            return HypothesisAction.SWITCH

        # Default to inspect if unclear
        return HypothesisAction.INSPECT

    def _determine_switch_target(
        self,
        bundle: Stage2EvidenceBundle,
        request_text: str,
        preferred_kind: str,
    ) -> Optional[str]:
        """Determine which hypothesis kind to switch to."""
        # If explicit preference given, use it
        if preferred_kind:
            kind_lower = preferred_kind.lower().strip()
            if kind_lower in ("direct", "proxy", "context"):
                return kind_lower

        # Parse from request text
        text_lower = request_text.lower()
        for kind in ("direct", "proxy", "context"):
            if kind in text_lower:
                return kind

        # Check for semantic hints
        if any(kw in text_lower for kw in ["exact", "literal", "strict"]):
            return "direct"
        if any(
            kw in text_lower for kw in ["similar", "alternative", "proxy", "fallback"]
        ):
            return "proxy"
        if any(
            kw in text_lower for kw in ["broader", "context", "scene", "surrounding"]
        ):
            return "context"

        return None

    def _extract_expansion_hints(self, request_text: str) -> List[str]:
        """Extract hints for hypothesis expansion from request text."""
        import re

        hints = []

        # Find quoted strings
        quoted = re.findall(r'"([^"]+)"', request_text)
        hints.extend(quoted)

        quoted_single = re.findall(r"'([^']+)'", request_text)
        hints.extend(quoted_single)

        # Deduplicate
        seen = set()
        result = []
        for hint in hints:
            hint = hint.strip()
            if hint and hint.lower() not in seen:
                seen.add(hint.lower())
                result.append(hint)

        return result

    def create_callback(
        self,
    ) -> Callable[[Stage2EvidenceBundle, Dict[str, Any]], Stage2ToolResult]:
        """Create a callback function for the Stage-2 agent.

        Returns:
            Callback compatible with Stage2DeepResearchAgent.hypothesis_callback
        """

        def callback(
            bundle: Stage2EvidenceBundle,
            request: Dict[str, Any],
        ) -> Stage2ToolResult:
            return self.handle_tool_request(bundle, request)

        return callback


def create_hypothesis_repair_callback(
    config: Optional[HypothesisRepairConfig] = None,
    stage1_reparse_callback: Optional[
        Callable[[Stage2EvidenceBundle, Dict[str, Any]], Stage2EvidenceBundle]
    ] = None,
) -> Callable[[Stage2EvidenceBundle, Dict[str, Any]], Stage2ToolResult]:
    """Convenience function to create a hypothesis repair callback.

    Args:
        config: Optional backend configuration
        stage1_reparse_callback: Optional callback to request Stage-1 re-parsing

    Returns:
        Callback function for Stage2DeepResearchAgent.hypothesis_callback

    Example:
        callback = create_hypothesis_repair_callback()
        agent = Stage2DeepResearchAgent(hypothesis_callback=callback)
    """
    backend = HypothesisRepairBackend(
        config=config, stage1_reparse_callback=stage1_reparse_callback
    )
    return backend.create_callback()
