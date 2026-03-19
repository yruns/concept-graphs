"""Unit tests for ablation configuration system.

Tests cover:
- YAML serialization/deserialization
- Config validation
- Preset configurations
- Ablation matrix generation
- BatchEvalConfig conversion
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from conceptgraph.evaluation.ablation_config import (
    AblationConfig,
    AgentConfig,
    EvaluationConfig,
    Stage1Config,
    Stage2Config,
    ToolConfig,
    generate_ablation_matrix,
    get_all_presets,
    get_preset_config,
    load_experiment_configs,
    save_ablation_matrix,
)

# =============================================================================
# ToolConfig Tests
# =============================================================================


class TestToolConfig:
    """Tests for ToolConfig model."""

    def test_default_all_enabled(self):
        """Test that all tools are enabled by default."""
        config = ToolConfig()
        assert config.request_more_views is True
        assert config.request_crops is True
        assert config.switch_or_expand_hypothesis is True
        assert config.inspect_stage1_metadata is True
        assert config.retrieve_object_context is True

    def test_enabled_tools_all(self):
        """Test enabled_tools returns all tools when enabled."""
        config = ToolConfig()
        enabled = config.enabled_tools()
        assert len(enabled) == 5
        assert "request_more_views" in enabled
        assert "request_crops" in enabled
        assert "switch_or_expand_hypothesis" in enabled

    def test_enabled_tools_partial(self):
        """Test enabled_tools with some tools disabled."""
        config = ToolConfig(request_crops=False, request_more_views=False)
        enabled = config.enabled_tools()
        assert "request_crops" not in enabled
        assert "request_more_views" not in enabled
        assert "switch_or_expand_hypothesis" in enabled

    def test_disabled_tools(self):
        """Test disabled_tools returns correct list."""
        config = ToolConfig(request_crops=False)
        disabled = config.disabled_tools()
        assert "request_crops" in disabled
        assert len(disabled) == 1

    def test_all_tools_disabled(self):
        """Test with all tools disabled."""
        config = ToolConfig(
            request_more_views=False,
            request_crops=False,
            switch_or_expand_hypothesis=False,
            inspect_stage1_metadata=False,
            retrieve_object_context=False,
        )
        assert len(config.enabled_tools()) == 0
        assert len(config.disabled_tools()) == 5


# =============================================================================
# AgentConfig Tests
# =============================================================================


class TestAgentConfig:
    """Tests for AgentConfig model."""

    def test_default_values(self):
        """Test default agent configuration values."""
        config = AgentConfig()
        assert config.max_turns == 6
        assert config.plan_mode == "brief"
        assert config.confidence_threshold == 0.4
        assert config.enable_uncertainty_stopping is True
        assert config.temperature == 0.1
        assert config.enable_subagents is True
        assert config.max_images == 6

    def test_max_turns_validation(self):
        """Test max_turns validation constraints."""
        # Valid values
        AgentConfig(max_turns=1)
        AgentConfig(max_turns=12)

        # Invalid values
        with pytest.raises(ValueError):
            AgentConfig(max_turns=0)
        with pytest.raises(ValueError):
            AgentConfig(max_turns=13)

    def test_plan_mode_validation(self):
        """Test plan_mode accepts only valid values."""
        AgentConfig(plan_mode="off")
        AgentConfig(plan_mode="brief")
        AgentConfig(plan_mode="full")

        with pytest.raises(ValueError):
            AgentConfig(plan_mode="invalid")

    def test_confidence_threshold_validation(self):
        """Test confidence_threshold validation."""
        AgentConfig(confidence_threshold=0.0)
        AgentConfig(confidence_threshold=1.0)

        with pytest.raises(ValueError):
            AgentConfig(confidence_threshold=-0.1)
        with pytest.raises(ValueError):
            AgentConfig(confidence_threshold=1.1)

    def test_temperature_validation(self):
        """Test temperature validation."""
        AgentConfig(temperature=0.0)
        AgentConfig(temperature=2.0)

        with pytest.raises(ValueError):
            AgentConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            AgentConfig(temperature=2.1)


# =============================================================================
# Stage1Config Tests
# =============================================================================


class TestStage1Config:
    """Tests for Stage1Config model."""

    def test_default_values(self):
        """Test default Stage 1 configuration."""
        config = Stage1Config()
        assert config.model == "gpt-5.2-2025-12-11"
        assert config.k == 3
        assert config.timeout_seconds == 60

    def test_k_validation(self):
        """Test k validation constraints."""
        Stage1Config(k=1)
        Stage1Config(k=10)

        with pytest.raises(ValueError):
            Stage1Config(k=0)
        with pytest.raises(ValueError):
            Stage1Config(k=11)


# =============================================================================
# Stage2Config Tests
# =============================================================================


class TestStage2Config:
    """Tests for Stage2Config model."""

    def test_default_values(self):
        """Test default Stage 2 configuration."""
        config = Stage2Config()
        assert config.enabled is True
        assert config.model == "gpt-5.2-2025-12-11"
        assert config.timeout_seconds == 120
        assert "genai-sg-og" in config.base_url

    def test_disabled_stage2(self):
        """Test Stage 2 can be disabled."""
        config = Stage2Config(enabled=False)
        assert config.enabled is False


# =============================================================================
# EvaluationConfig Tests
# =============================================================================


class TestEvaluationConfig:
    """Tests for EvaluationConfig model."""

    def test_default_values(self):
        """Test default evaluation configuration."""
        config = EvaluationConfig()
        assert config.max_workers == 4
        assert config.batch_size == 10
        assert config.checkpoint_interval == 10
        assert config.max_samples is None
        assert config.skip_samples == 0

    def test_limits(self):
        """Test evaluation limit settings."""
        config = EvaluationConfig(max_samples=100, skip_samples=10)
        assert config.max_samples == 100
        assert config.skip_samples == 10


# =============================================================================
# AblationConfig Tests
# =============================================================================


class TestAblationConfig:
    """Tests for main AblationConfig model."""

    def test_minimal_config(self):
        """Test minimal required configuration."""
        config = AblationConfig(name="test")
        assert config.name == "test"
        assert config.description == ""
        assert isinstance(config.tools, ToolConfig)
        assert isinstance(config.agent, AgentConfig)

    def test_full_config(self):
        """Test full configuration with all fields."""
        config = AblationConfig(
            name="test_full",
            description="Test configuration",
            tools=ToolConfig(request_crops=False),
            agent=AgentConfig(max_turns=4),
            stage1=Stage1Config(k=5),
            stage2=Stage2Config(enabled=True),
            evaluation=EvaluationConfig(max_workers=8),
            tags=["test", "ablation"],
            parent="full",
        )
        assert config.name == "test_full"
        assert config.description == "Test configuration"
        assert config.tools.request_crops is False
        assert config.agent.max_turns == 4
        assert config.stage1.k == 5
        assert config.evaluation.max_workers == 8
        assert "test" in config.tags
        assert config.parent == "full"

    def test_name_required(self):
        """Test that name is required."""
        with pytest.raises(ValueError):
            AblationConfig()

    def test_name_not_empty(self):
        """Test that name cannot be empty."""
        with pytest.raises(ValueError):
            AblationConfig(name="")


# =============================================================================
# Ablation Tag Generation Tests
# =============================================================================


class TestAblationTag:
    """Tests for ablation tag generation."""

    def test_full_tag(self):
        """Test full config generates 'full' tag."""
        config = AblationConfig(name="test")
        assert config.get_ablation_tag() == "full"

    def test_stage1_only_tag(self):
        """Test stage1_only config generates correct tag."""
        config = AblationConfig(
            name="test",
            stage2=Stage2Config(enabled=False),
        )
        assert config.get_ablation_tag() == "stage1_only"

    def test_no_views_tag(self):
        """Test config without views generates 'no_views' tag."""
        config = AblationConfig(
            name="test",
            tools=ToolConfig(request_more_views=False),
        )
        assert config.get_ablation_tag() == "no_views"

    def test_no_crops_tag(self):
        """Test config without crops generates 'no_crops' tag."""
        config = AblationConfig(
            name="test",
            tools=ToolConfig(request_crops=False),
        )
        assert config.get_ablation_tag() == "no_crops"

    def test_no_repair_tag(self):
        """Test config without repair generates 'no_repair' tag."""
        config = AblationConfig(
            name="test",
            tools=ToolConfig(switch_or_expand_hypothesis=False),
        )
        assert config.get_ablation_tag() == "no_repair"

    def test_no_uncertainty_tag(self):
        """Test config without uncertainty generates 'no_uncertainty' tag."""
        config = AblationConfig(
            name="test",
            agent=AgentConfig(enable_uncertainty_stopping=False),
        )
        assert config.get_ablation_tag() == "no_uncertainty"

    def test_oneshot_tag(self):
        """Test one-shot config generates 'oneshot' tag."""
        config = AblationConfig(
            name="test",
            agent=AgentConfig(max_turns=1),
        )
        assert config.get_ablation_tag() == "oneshot"

    def test_combined_tags(self):
        """Test config with multiple ablations generates combined tag."""
        config = AblationConfig(
            name="test",
            tools=ToolConfig(request_crops=False, request_more_views=False),
        )
        tag = config.get_ablation_tag()
        assert "no_views" in tag
        assert "no_crops" in tag


# =============================================================================
# YAML Serialization Tests
# =============================================================================


class TestYAMLSerialization:
    """Tests for YAML serialization/deserialization."""

    def test_to_yaml_string(self):
        """Test serialization to YAML string."""
        config = AblationConfig(name="test", description="Test config")
        yaml_str = config.to_yaml()
        assert "name: test" in yaml_str
        assert "description: Test config" in yaml_str

    def test_to_yaml_file(self):
        """Test serialization to YAML file."""
        config = AblationConfig(name="test_file")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yaml"
            config.to_yaml(path)
            assert path.exists()
            # Read back and verify
            loaded = AblationConfig.from_yaml(path)
            assert loaded.name == "test_file"

    def test_from_yaml_file(self):
        """Test deserialization from YAML file."""
        yaml_content = """
name: test_load
description: Test loading from YAML
tools:
  request_crops: false
agent:
  max_turns: 4
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml_content)
            config = AblationConfig.from_yaml(path)
            assert config.name == "test_load"
            assert config.description == "Test loading from YAML"
            assert config.tools.request_crops is False
            assert config.agent.max_turns == 4

    def test_from_yaml_string(self):
        """Test deserialization from YAML string."""
        yaml_str = """
name: test_string
tools:
  request_more_views: false
"""
        config = AblationConfig.from_yaml_string(yaml_str)
        assert config.name == "test_string"
        assert config.tools.request_more_views is False

    def test_from_yaml_file_not_found(self):
        """Test error on non-existent file."""
        with pytest.raises(FileNotFoundError):
            AblationConfig.from_yaml("/nonexistent/path.yaml")

    def test_from_yaml_empty_file(self):
        """Test error on empty YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.yaml"
            path.write_text("")
            with pytest.raises(ValueError, match="Empty"):
                AblationConfig.from_yaml(path)

    def test_from_yaml_invalid_schema(self):
        """Test error on invalid config schema."""
        yaml_str = """
name: test
agent:
  max_turns: 100
"""
        with pytest.raises(ValueError):
            AblationConfig.from_yaml_string(yaml_str)

    def test_round_trip(self):
        """Test YAML round-trip preserves values."""
        original = AblationConfig(
            name="round_trip",
            description="Round trip test",
            tools=ToolConfig(request_crops=False, request_more_views=False),
            agent=AgentConfig(max_turns=4, confidence_threshold=0.6),
            stage1=Stage1Config(k=5),
            tags=["test", "round_trip"],
        )
        yaml_str = original.to_yaml()
        loaded = AblationConfig.from_yaml_string(yaml_str)

        assert loaded.name == original.name
        assert loaded.description == original.description
        assert loaded.tools.request_crops == original.tools.request_crops
        assert loaded.tools.request_more_views == original.tools.request_more_views
        assert loaded.agent.max_turns == original.agent.max_turns
        assert loaded.agent.confidence_threshold == original.agent.confidence_threshold
        assert loaded.stage1.k == original.stage1.k


# =============================================================================
# BatchEvalConfig Conversion Tests
# =============================================================================


class TestBatchEvalConfigConversion:
    """Tests for conversion to BatchEvalConfig format."""

    def test_basic_conversion(self):
        """Test basic conversion to BatchEvalConfig dict."""
        config = AblationConfig(name="test")
        batch_dict = config.to_batch_eval_config()

        assert batch_dict["run_id"] == "test"
        assert batch_dict["stage2_enabled"] is True
        assert batch_dict["enable_tool_request_crops"] is True
        assert batch_dict["enable_uncertainty_stopping"] is True

    def test_tool_ablation_conversion(self):
        """Test tool ablation reflects in conversion."""
        config = AblationConfig(
            name="test",
            tools=ToolConfig(request_crops=False, request_more_views=False),
        )
        batch_dict = config.to_batch_eval_config()

        assert batch_dict["enable_tool_request_crops"] is False
        assert batch_dict["enable_tool_request_more_views"] is False
        assert batch_dict["enable_tool_hypothesis_repair"] is True

    def test_agent_params_conversion(self):
        """Test agent parameters reflected in conversion."""
        config = AblationConfig(
            name="test",
            agent=AgentConfig(
                max_turns=4,
                confidence_threshold=0.6,
                enable_uncertainty_stopping=False,
            ),
        )
        batch_dict = config.to_batch_eval_config()

        assert batch_dict["stage2_max_turns"] == 4
        assert batch_dict["confidence_threshold"] == 0.6
        assert batch_dict["enable_uncertainty_stopping"] is False

    def test_stage1_params_conversion(self):
        """Test Stage 1 parameters reflected in conversion."""
        config = AblationConfig(
            name="test",
            stage1=Stage1Config(model="gpt-4", k=5),
        )
        batch_dict = config.to_batch_eval_config()

        assert batch_dict["stage1_model"] == "gpt-4"
        assert batch_dict["stage1_k"] == 5

    def test_evaluation_params_conversion(self):
        """Test evaluation parameters reflected in conversion."""
        config = AblationConfig(
            name="test",
            evaluation=EvaluationConfig(
                max_workers=8,
                max_samples=100,
                skip_samples=10,
            ),
        )
        batch_dict = config.to_batch_eval_config()

        assert batch_dict["max_workers"] == 8
        assert batch_dict["max_samples"] == 100
        assert batch_dict["skip_samples"] == 10


# =============================================================================
# Preset Configuration Tests
# =============================================================================


class TestPresetConfigs:
    """Tests for preset configurations."""

    def test_get_all_presets(self):
        """Test get_all_presets returns expected presets."""
        presets = get_all_presets()
        expected = [
            "full",
            "stage1_only",
            "oneshot",
            "no_views",
            "no_crops",
            "no_repair",
            "no_uncertainty",
            "minimal_tools",
        ]
        for name in expected:
            assert name in presets
            assert isinstance(presets[name], AblationConfig)

    def test_get_preset_full(self):
        """Test full preset configuration."""
        config = get_preset_config("full")
        assert config.name == "full"
        assert config.stage2.enabled is True
        assert config.get_ablation_tag() == "full"

    def test_get_preset_stage1_only(self):
        """Test stage1_only preset disables Stage 2."""
        config = get_preset_config("stage1_only")
        assert config.name == "stage1_only"
        assert config.stage2.enabled is False
        assert config.get_ablation_tag() == "stage1_only"

    def test_get_preset_oneshot(self):
        """Test oneshot preset limits to single turn."""
        config = get_preset_config("oneshot")
        assert config.name == "oneshot"
        assert config.agent.max_turns == 1
        assert config.tools.request_crops is False
        assert config.tools.request_more_views is False
        assert config.tools.switch_or_expand_hypothesis is False
        # Tag reflects all disabled tools + oneshot marker
        assert config.get_ablation_tag() == "no_views_no_crops_no_repair_oneshot"

    def test_get_preset_no_crops(self):
        """Test no_crops preset disables crop tool."""
        config = get_preset_config("no_crops")
        assert config.tools.request_crops is False
        assert config.tools.request_more_views is True
        assert config.get_ablation_tag() == "no_crops"

    def test_get_preset_no_uncertainty(self):
        """Test no_uncertainty preset disables uncertainty stopping."""
        config = get_preset_config("no_uncertainty")
        assert config.agent.enable_uncertainty_stopping is False
        assert config.get_ablation_tag() == "no_uncertainty"

    def test_get_preset_minimal_tools(self):
        """Test minimal_tools preset enables only inspection tools."""
        config = get_preset_config("minimal_tools")
        assert config.tools.request_crops is False
        assert config.tools.request_more_views is False
        assert config.tools.switch_or_expand_hypothesis is False
        assert config.tools.inspect_stage1_metadata is True
        assert config.tools.retrieve_object_context is True

    def test_get_preset_invalid(self):
        """Test error on invalid preset name."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_config("invalid_preset")


# =============================================================================
# Ablation Matrix Generation Tests
# =============================================================================


class TestAblationMatrix:
    """Tests for ablation matrix generation."""

    def test_generate_matrix_default(self):
        """Test matrix generation with default base config."""
        matrix = generate_ablation_matrix()

        # Should include key ablation conditions
        assert "full" in matrix
        assert "oneshot" in matrix
        assert "stage1_only" in matrix
        assert "no_views" in matrix
        assert "no_crops" in matrix
        assert "no_repair" in matrix
        assert "no_uncertainty" in matrix

    def test_generate_matrix_configs_are_valid(self):
        """Test all generated configs are valid."""
        matrix = generate_ablation_matrix()

        for name, config in matrix.items():
            assert isinstance(config, AblationConfig)
            assert config.name == name
            # Should all have parent set except full
            if name != "full":
                assert config.parent == "full"

    def test_generate_matrix_tags_correct(self):
        """Test ablation tags match config names."""
        matrix = generate_ablation_matrix()

        for name, config in matrix.items():
            tag = config.get_ablation_tag()
            if name == "full":
                assert tag == "full"
            elif name == "oneshot":
                assert "oneshot" in tag
            elif name == "stage1_only":
                assert tag == "stage1_only"
            elif name.startswith("no_"):
                assert name in tag or name.replace("_", "") in tag

    def test_generate_matrix_custom_base(self):
        """Test matrix generation with custom base config."""
        base = AblationConfig(
            name="custom_base",
            agent=AgentConfig(max_turns=4, confidence_threshold=0.5),
        )
        matrix = generate_ablation_matrix(base)

        # Check all derived configs inherit base agent settings
        for name, config in matrix.items():
            if name != "oneshot" and name != "stage1_only":
                # Confidence threshold should be inherited
                assert config.agent.confidence_threshold == 0.5


# =============================================================================
# File I/O Tests
# =============================================================================


class TestFileIO:
    """Tests for configuration file I/O."""

    def test_load_experiment_configs(self):
        """Test loading configs from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test configs
            config1 = AblationConfig(name="test1")
            config2 = AblationConfig(name="test2")
            config1.to_yaml(Path(tmpdir) / "test1.yaml")
            config2.to_yaml(Path(tmpdir) / "test2.yml")

            configs = load_experiment_configs(tmpdir)
            assert "test1" in configs
            assert "test2" in configs

    def test_load_experiment_configs_empty_dir(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = load_experiment_configs(tmpdir)
            assert len(configs) == 0

    def test_load_experiment_configs_nonexistent_dir(self):
        """Test loading from non-existent directory."""
        configs = load_experiment_configs("/nonexistent/path")
        assert len(configs) == 0

    def test_save_ablation_matrix(self):
        """Test saving ablation matrix to directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            matrix = generate_ablation_matrix()
            save_ablation_matrix(matrix, tmpdir)

            # Check files were created
            yaml_files = list(Path(tmpdir).glob("*.yaml"))
            assert len(yaml_files) == len(matrix)

            # Verify files can be loaded back
            loaded = load_experiment_configs(tmpdir)
            assert set(loaded.keys()) == set(matrix.keys())


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_create_run_ablation_workflow(self):
        """Test complete workflow: create config, save, load, convert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create custom config
            config = AblationConfig(
                name="integration_test",
                description="Integration test config",
                tools=ToolConfig(request_crops=False),
                agent=AgentConfig(max_turns=4),
            )

            # Save to file
            config_path = Path(tmpdir) / "test_config.yaml"
            config.to_yaml(config_path)

            # Load from file
            loaded = AblationConfig.from_yaml(config_path)

            # Convert to batch eval config
            batch_config = loaded.to_batch_eval_config()

            # Verify values preserved
            assert batch_config["enable_tool_request_crops"] is False
            assert batch_config["stage2_max_turns"] == 4
            assert loaded.get_ablation_tag() == "no_crops"

    def test_preset_to_batch_eval_workflow(self):
        """Test using preset to configure batch evaluation."""
        # Get preset
        preset = get_preset_config("oneshot")

        # Convert to batch eval config
        batch_config = preset.to_batch_eval_config()

        # Verify one-shot specific settings
        assert batch_config["stage2_max_turns"] == 1
        assert batch_config["enable_tool_request_crops"] is False
        assert batch_config["enable_tool_request_more_views"] is False

    def test_generate_save_load_matrix(self):
        """Test generate matrix, save all, load all."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate matrix
            original_matrix = generate_ablation_matrix()

            # Save all configs
            save_ablation_matrix(original_matrix, tmpdir)

            # Load all configs
            loaded_matrix = load_experiment_configs(tmpdir)

            # Verify all configs preserved
            assert set(loaded_matrix.keys()) == set(original_matrix.keys())

            # Verify specific config values
            orig_crops = original_matrix["no_crops"]
            loaded_crops = loaded_matrix["no_crops"]
            assert loaded_crops.tools.request_crops == orig_crops.tools.request_crops
