"""Ablation configuration system for two-stage 3D scene understanding evaluation.

This module provides YAML-based configuration for systematic ablation studies,
enabling fine-grained control over:
- Individual tool enable/disable
- Agent parameters (confidence thresholds, max turns, plan mode)
- Stage 1/2 model selection
- Batch evaluation settings

Academic Support:
- Standardized ablation configs ensure reproducible experiments
- Tool-level toggles enable isolated component analysis
- Preset configurations cover common experimental conditions
- Serializable configs support experiment tracking and reproduction

Example YAML configuration:
```yaml
name: no_crops
description: Ablation without crop tool

tools:
  request_more_views: true
  request_crops: false
  switch_or_expand_hypothesis: true

agent:
  max_turns: 6
  plan_mode: brief
  confidence_threshold: 0.4
  enable_uncertainty_stopping: true

stage1:
  model: gpt-5.2-2025-12-11
  k: 3

stage2:
  model: gpt-5.2-2025-12-11
  enabled: true
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from loguru import logger
from pydantic import BaseModel, Field, field_validator

# =============================================================================
# Configuration Models (Pydantic for validation)
# =============================================================================


class ToolConfig(BaseModel):
    """Configuration for Stage 2 agent tools.

    Each tool can be individually enabled/disabled to support
    component-level ablation studies.
    """

    request_more_views: bool = Field(
        default=True,
        description="Enable the request_more_views tool for additional keyframe retrieval",
    )
    request_crops: bool = Field(
        default=True,
        description="Enable the request_crops tool for object-centric cropping",
    )
    switch_or_expand_hypothesis: bool = Field(
        default=True,
        description="Enable the switch_or_expand_hypothesis tool for hypothesis repair",
    )
    inspect_stage1_metadata: bool = Field(
        default=True,
        description="Enable the inspect_stage1_metadata tool for Stage 1 introspection",
    )
    retrieve_object_context: bool = Field(
        default=True,
        description="Enable the retrieve_object_context tool for scene context lookup",
    )

    def enabled_tools(self) -> List[str]:
        """Return list of enabled tool names."""
        enabled = []
        for tool_name in [
            "request_more_views",
            "request_crops",
            "switch_or_expand_hypothesis",
            "inspect_stage1_metadata",
            "retrieve_object_context",
        ]:
            if getattr(self, tool_name, False):
                enabled.append(tool_name)
        return enabled

    def disabled_tools(self) -> List[str]:
        """Return list of disabled tool names."""
        all_tools = {
            "request_more_views",
            "request_crops",
            "switch_or_expand_hypothesis",
            "inspect_stage1_metadata",
            "retrieve_object_context",
        }
        return list(all_tools - set(self.enabled_tools()))


class AgentConfig(BaseModel):
    """Configuration for Stage 2 agent behavior.

    Controls high-level agent parameters that affect
    reasoning depth and uncertainty handling.
    """

    max_turns: int = Field(
        default=6,
        ge=1,
        le=12,
        description="Maximum reasoning turns before forced completion",
    )
    plan_mode: Literal["off", "brief", "full"] = Field(
        default="brief",
        description="Planning mode: off (no planning), brief (2-4 items), full (explicit decomposition)",
    )
    confidence_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for task completion",
    )
    enable_uncertainty_stopping: bool = Field(
        default=True,
        description="Enable uncertainty-aware stopping when evidence is insufficient",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Model temperature for response generation",
    )
    enable_subagents: bool = Field(
        default=True,
        description="Enable subagent delegation for complex tasks (FULL mode only)",
    )
    max_images: int = Field(
        default=6,
        ge=1,
        le=12,
        description="Maximum number of keyframe images to inject",
    )
    image_max_size: int = Field(
        default=900,
        ge=256,
        le=2048,
        description="Maximum image dimension for resizing",
    )


class Stage1Config(BaseModel):
    """Configuration for Stage 1 keyframe retrieval."""

    model: str = Field(
        default="gpt-5.2-2025-12-11",
        description="Model for query parsing and hypothesis generation",
    )
    k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of keyframes to retrieve",
    )
    timeout_seconds: int = Field(
        default=60,
        ge=10,
        description="Timeout for Stage 1 retrieval",
    )


class Stage2Config(BaseModel):
    """Configuration for Stage 2 VLM agent."""

    enabled: bool = Field(
        default=True,
        description="Whether to run Stage 2 agent (false for Stage 1 only baseline)",
    )
    model: str = Field(
        default="gpt-5.2-2025-12-11",
        description="Model for VLM agent reasoning",
    )
    timeout_seconds: int = Field(
        default=120,
        ge=30,
        description="Timeout for Stage 2 agent run",
    )
    base_url: str = Field(
        default="https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl",
        description="API base URL for Stage 2 model",
    )


class EvaluationConfig(BaseModel):
    """Configuration for batch evaluation settings."""

    max_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of parallel workers",
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Batch size for progress tracking",
    )
    checkpoint_interval: int = Field(
        default=10,
        ge=1,
        description="Save checkpoint every N samples",
    )
    max_samples: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum samples to evaluate (None for all)",
    )
    skip_samples: int = Field(
        default=0,
        ge=0,
        description="Number of initial samples to skip",
    )


class AblationConfig(BaseModel):
    """Complete ablation configuration for evaluation experiments.

    This is the main configuration class that composes all sub-configs
    and provides YAML serialization/deserialization.
    """

    name: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for this ablation configuration",
    )
    description: str = Field(
        default="",
        description="Human-readable description of this ablation",
    )
    tools: ToolConfig = Field(
        default_factory=ToolConfig,
        description="Tool enable/disable configuration",
    )
    agent: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Agent behavior configuration",
    )
    stage1: Stage1Config = Field(
        default_factory=Stage1Config,
        description="Stage 1 retrieval configuration",
    )
    stage2: Stage2Config = Field(
        default_factory=Stage2Config,
        description="Stage 2 agent configuration",
    )
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Batch evaluation configuration",
    )

    # Metadata for experiment tracking
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for experiment categorization",
    )
    parent: Optional[str] = Field(
        default=None,
        description="Parent config name if this is a derived configuration",
    )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AblationConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Parsed AblationConfig instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            yaml.YAMLError: If the YAML is invalid.
            ValueError: If the config is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Empty configuration file: {path}")

        return cls.model_validate(data)

    @classmethod
    def from_yaml_string(cls, yaml_string: str) -> "AblationConfig":
        """Load configuration from a YAML string.

        Args:
            yaml_string: YAML configuration as a string.

        Returns:
            Parsed AblationConfig instance.
        """
        data = yaml.safe_load(yaml_string)
        if data is None:
            raise ValueError("Empty configuration string")
        return cls.model_validate(data)

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        """Serialize configuration to YAML format.

        Args:
            path: Optional path to write YAML file.

        Returns:
            YAML string representation.
        """
        data = self.model_dump(exclude_none=True, exclude_defaults=False)
        yaml_string = yaml.dump(data, default_flow_style=False, sort_keys=False)

        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(yaml_string)
            logger.info(f"Saved ablation config to {path}")

        return yaml_string

    def get_ablation_tag(self) -> str:
        """Generate a standardized ablation tag for result grouping.

        The tag encodes key ablation variables in a compact format
        for use in result files and tables.
        """
        if not self.stage2.enabled:
            return "stage1_only"

        tags = []

        # Tool ablations
        if not self.tools.request_more_views:
            tags.append("no_views")
        if not self.tools.request_crops:
            tags.append("no_crops")
        if not self.tools.switch_or_expand_hypothesis:
            tags.append("no_repair")

        # Agent ablations
        if not self.agent.enable_uncertainty_stopping:
            tags.append("no_uncertainty")
        if self.agent.max_turns == 1:
            tags.append("oneshot")

        return "_".join(tags) if tags else "full"

    def to_batch_eval_config(self) -> Dict[str, Any]:
        """Convert to BatchEvalConfig-compatible dictionary.

        Returns a dictionary that can be used to construct BatchEvalConfig
        or be passed directly to BatchEvaluator.
        """
        return {
            "run_id": self.name,
            "benchmark_name": self.name,
            # Parallelism
            "max_workers": self.evaluation.max_workers,
            "batch_size": self.evaluation.batch_size,
            "checkpoint_interval": self.evaluation.checkpoint_interval,
            # Stage 1
            "stage1_model": self.stage1.model,
            "stage1_k": self.stage1.k,
            "stage1_timeout_seconds": self.stage1.timeout_seconds,
            # Stage 2
            "stage2_enabled": self.stage2.enabled,
            "stage2_model": self.stage2.model,
            "stage2_max_turns": self.agent.max_turns,
            "stage2_plan_mode": self.agent.plan_mode,
            "stage2_timeout_seconds": self.stage2.timeout_seconds,
            # Tool ablations
            "enable_tool_request_more_views": self.tools.request_more_views,
            "enable_tool_request_crops": self.tools.request_crops,
            "enable_tool_hypothesis_repair": self.tools.switch_or_expand_hypothesis,
            # Uncertainty
            "enable_uncertainty_stopping": self.agent.enable_uncertainty_stopping,
            "confidence_threshold": self.agent.confidence_threshold,
            # Limits
            "max_samples": self.evaluation.max_samples,
            "skip_samples": self.evaluation.skip_samples,
        }


# =============================================================================
# Preset Configurations for Common Ablation Studies
# =============================================================================


def get_preset_config(preset_name: str) -> AblationConfig:
    """Get a preset ablation configuration by name.

    Available presets:
    - full: Full Stage 2 agent with all tools enabled
    - stage1_only: Stage 1 retrieval only (no VLM agent)
    - oneshot: One-shot VLM (no tool calls allowed)
    - no_views: Without request_more_views tool
    - no_crops: Without request_crops tool
    - no_repair: Without hypothesis_repair tool
    - no_uncertainty: Without uncertainty stopping
    - minimal_tools: Only basic inspection tools

    Args:
        preset_name: Name of the preset configuration.

    Returns:
        AblationConfig instance.

    Raises:
        ValueError: If the preset name is not recognized.
    """
    presets = get_all_presets()
    if preset_name not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available}"
        )
    return presets[preset_name]


def get_all_presets() -> Dict[str, AblationConfig]:
    """Get all available preset configurations.

    Returns:
        Dictionary mapping preset names to AblationConfig instances.
    """
    return {
        "full": _preset_full(),
        "stage1_only": _preset_stage1_only(),
        "oneshot": _preset_oneshot(),
        "no_views": _preset_no_views(),
        "no_crops": _preset_no_crops(),
        "no_repair": _preset_no_repair(),
        "no_uncertainty": _preset_no_uncertainty(),
        "minimal_tools": _preset_minimal_tools(),
    }


def _preset_full() -> AblationConfig:
    """Full Stage 2 agent with all tools enabled."""
    return AblationConfig(
        name="full",
        description="Full Stage 2 agent with all tools enabled",
        tags=["baseline", "full"],
    )


def _preset_stage1_only() -> AblationConfig:
    """Stage 1 retrieval only (no VLM agent)."""
    return AblationConfig(
        name="stage1_only",
        description="Stage 1 retrieval only (no VLM agent)",
        stage2=Stage2Config(enabled=False),
        tags=["baseline", "ablation"],
    )


def _preset_oneshot() -> AblationConfig:
    """One-shot VLM without tool calls."""
    return AblationConfig(
        name="oneshot",
        description="One-shot VLM inference without tool calls",
        agent=AgentConfig(max_turns=1, enable_subagents=False),
        tools=ToolConfig(
            request_more_views=False,
            request_crops=False,
            switch_or_expand_hypothesis=False,
            # Keep inspection tools for context
            inspect_stage1_metadata=True,
            retrieve_object_context=True,
        ),
        tags=["baseline", "ablation"],
    )


def _preset_no_views() -> AblationConfig:
    """Without request_more_views tool."""
    return AblationConfig(
        name="no_views",
        description="Ablation: without request_more_views tool",
        tools=ToolConfig(request_more_views=False),
        tags=["ablation", "tool_ablation"],
    )


def _preset_no_crops() -> AblationConfig:
    """Without request_crops tool."""
    return AblationConfig(
        name="no_crops",
        description="Ablation: without request_crops tool",
        tools=ToolConfig(request_crops=False),
        tags=["ablation", "tool_ablation"],
    )


def _preset_no_repair() -> AblationConfig:
    """Without hypothesis_repair tool."""
    return AblationConfig(
        name="no_repair",
        description="Ablation: without hypothesis_repair tool",
        tools=ToolConfig(switch_or_expand_hypothesis=False),
        tags=["ablation", "tool_ablation"],
    )


def _preset_no_uncertainty() -> AblationConfig:
    """Without uncertainty stopping."""
    return AblationConfig(
        name="no_uncertainty",
        description="Ablation: without uncertainty-aware stopping",
        agent=AgentConfig(enable_uncertainty_stopping=False),
        tags=["ablation", "agent_ablation"],
    )


def _preset_minimal_tools() -> AblationConfig:
    """Only basic inspection tools, no evidence expansion."""
    return AblationConfig(
        name="minimal_tools",
        description="Only inspect/context tools, no evidence expansion",
        tools=ToolConfig(
            request_more_views=False,
            request_crops=False,
            switch_or_expand_hypothesis=False,
            inspect_stage1_metadata=True,
            retrieve_object_context=True,
        ),
        tags=["ablation", "tool_ablation"],
    )


# =============================================================================
# Experiment Configuration Helpers
# =============================================================================


def load_experiment_configs(
    config_dir: Union[str, Path],
) -> Dict[str, AblationConfig]:
    """Load all YAML configurations from a directory.

    Args:
        config_dir: Directory containing YAML config files.

    Returns:
        Dictionary mapping config names to AblationConfig instances.
    """
    config_dir = Path(config_dir)
    if not config_dir.exists():
        logger.warning(f"Config directory does not exist: {config_dir}")
        return {}

    configs = {}
    for yaml_file in config_dir.glob("*.yaml"):
        try:
            config = AblationConfig.from_yaml(yaml_file)
            configs[config.name] = config
            logger.debug(f"Loaded config: {config.name} from {yaml_file}")
        except Exception as e:
            logger.error(f"Failed to load {yaml_file}: {e}")

    for yml_file in config_dir.glob("*.yml"):
        try:
            config = AblationConfig.from_yaml(yml_file)
            configs[config.name] = config
            logger.debug(f"Loaded config: {config.name} from {yml_file}")
        except Exception as e:
            logger.error(f"Failed to load {yml_file}: {e}")

    logger.info(f"Loaded {len(configs)} configs from {config_dir}")
    return configs


def generate_ablation_matrix(
    base_config: Optional[AblationConfig] = None,
) -> Dict[str, AblationConfig]:
    """Generate a comprehensive ablation matrix from a base config.

    Creates configurations varying one component at a time,
    useful for systematic ablation studies.

    Args:
        base_config: Base configuration (defaults to full preset).

    Returns:
        Dictionary mapping ablation names to AblationConfig instances.
    """
    if base_config is None:
        base_config = get_preset_config("full")

    matrix = {"full": base_config}

    # Tool ablations
    for tool in ["request_more_views", "request_crops", "switch_or_expand_hypothesis"]:
        ablation_name = f"no_{tool.replace('_', '').replace('request', '').replace('switch_or_expand', 'repair')}"
        if tool == "request_more_views":
            ablation_name = "no_views"
        elif tool == "request_crops":
            ablation_name = "no_crops"
        elif tool == "switch_or_expand_hypothesis":
            ablation_name = "no_repair"

        tools_dict = base_config.tools.model_dump()
        tools_dict[tool] = False

        matrix[ablation_name] = AblationConfig(
            name=ablation_name,
            description=f"Ablation: without {tool}",
            tools=ToolConfig.model_validate(tools_dict),
            agent=base_config.agent.model_copy(),
            stage1=base_config.stage1.model_copy(),
            stage2=base_config.stage2.model_copy(),
            evaluation=base_config.evaluation.model_copy(),
            tags=["ablation", "tool_ablation"],
            parent=base_config.name,
        )

    # One-shot ablation
    matrix["oneshot"] = AblationConfig(
        name="oneshot",
        description="One-shot VLM without multi-turn reasoning",
        tools=ToolConfig(
            request_more_views=False,
            request_crops=False,
            switch_or_expand_hypothesis=False,
        ),
        agent=AgentConfig(
            max_turns=1,
            enable_subagents=False,
            plan_mode="off",
            confidence_threshold=base_config.agent.confidence_threshold,
            enable_uncertainty_stopping=base_config.agent.enable_uncertainty_stopping,
        ),
        stage1=base_config.stage1.model_copy(),
        stage2=base_config.stage2.model_copy(),
        evaluation=base_config.evaluation.model_copy(),
        tags=["ablation", "turns_ablation"],
        parent=base_config.name,
    )

    # Stage 1 only
    matrix["stage1_only"] = AblationConfig(
        name="stage1_only",
        description="Stage 1 retrieval only (no VLM agent)",
        stage1=base_config.stage1.model_copy(),
        stage2=Stage2Config(enabled=False),
        evaluation=base_config.evaluation.model_copy(),
        tags=["baseline", "ablation"],
        parent=base_config.name,
    )

    # Uncertainty ablation
    matrix["no_uncertainty"] = AblationConfig(
        name="no_uncertainty",
        description="Ablation: without uncertainty-aware stopping",
        tools=base_config.tools.model_copy(),
        agent=AgentConfig(
            max_turns=base_config.agent.max_turns,
            plan_mode=base_config.agent.plan_mode,
            confidence_threshold=base_config.agent.confidence_threshold,
            enable_uncertainty_stopping=False,
            enable_subagents=base_config.agent.enable_subagents,
        ),
        stage1=base_config.stage1.model_copy(),
        stage2=base_config.stage2.model_copy(),
        evaluation=base_config.evaluation.model_copy(),
        tags=["ablation", "uncertainty_ablation"],
        parent=base_config.name,
    )

    return matrix


def save_ablation_matrix(
    configs: Dict[str, AblationConfig],
    output_dir: Union[str, Path],
) -> None:
    """Save a set of ablation configurations to a directory.

    Args:
        configs: Dictionary mapping names to AblationConfig instances.
        output_dir: Directory to write YAML files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, config in configs.items():
        config.to_yaml(output_dir / f"{name}.yaml")

    logger.info(f"Saved {len(configs)} ablation configs to {output_dir}")
