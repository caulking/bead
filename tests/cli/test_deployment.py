"""Integration tests for bead.cli.deployment CLI commands.

Tests all deployment commands to ensure they:
1. Generate valid jsPsych experiments
2. Handle distribution strategies correctly
3. Create proper output directory structure
4. Integrate correctly with core bead.deployment utilities
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bead.cli.deployment import deployment, export_jatos, generate, validate
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, PresentationSpec, TaskSpec
from bead.lists import ExperimentList


@pytest.fixture
def runner() -> CliRunner:
    """Create Click test runner."""
    return CliRunner()


@pytest.fixture
def sample_template() -> ItemTemplate:
    """Create a sample item template."""
    return ItemTemplate(
        name="test_template",
        description="Test item template",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(
            prompt="How natural is this sentence?",
            scale_bounds=(1, 7),
            scale_labels={1: "Very unnatural", 7: "Very natural"},
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )


@pytest.fixture
def sample_items_file(tmp_path: Path, sample_template: ItemTemplate) -> Path:
    """Create a sample items JSONL file."""
    items_file = tmp_path / "items.jsonl"

    # Create 10 test items
    items = []
    for i in range(10):
        item = Item(
            item_template_id=sample_template.id,
            rendered_elements={"text": f"This is test sentence number {i + 1}."},
            item_metadata={
                "condition": "A" if i % 2 == 0 else "B",
                "item_number": i,
                "scale_min": 1,
                "scale_max": 7,
            },
        )
        items.append(item.model_dump_json() + "\n")

    items_file.write_text("".join(items))
    return items_file


@pytest.fixture
def sample_lists_dir(tmp_path: Path, sample_items_file: Path) -> Path:
    """Create a sample lists directory with list files."""
    lists_dir = tmp_path / "lists"
    lists_dir.mkdir()

    # Read items to get their IDs
    items_data = [Item.model_validate_json(line) for line in sample_items_file.read_text().strip().split("\n")]
    item_ids = [item.id for item in items_data]

    # Create 3 lists
    for list_num in range(3):
        exp_list = ExperimentList(
            name=f"list_{list_num}",
            list_number=list_num,
        )

        # Add subset of items to each list (different items per list)
        start_idx = list_num * 3
        end_idx = start_idx + 4
        for item_id in item_ids[start_idx:end_idx]:
            exp_list.add_item(item_id)

        # Save list
        list_file = lists_dir / f"list_{list_num}.jsonl"
        list_file.write_text(exp_list.model_dump_json() + "\n")

    return lists_dir


# ==================== Generate Command Tests ====================


class TestGenerateCommand:
    """Test deployment generate command."""

    def test_generate_with_balanced_strategy(
        self,
        runner: CliRunner,
        sample_lists_dir: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test generate command with balanced distribution strategy."""
        output_dir = tmp_path / "experiment"

        result = runner.invoke(
            generate,
            [
                str(sample_lists_dir),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type", "likert_rating",
                "--title", "Test Experiment",
                "--distribution-strategy", "balanced",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Verify directory structure
        assert output_dir.exists()
        assert (output_dir / "index.html").exists()
        assert (output_dir / "css" / "experiment.css").exists()
        assert (output_dir / "js" / "experiment.js").exists()
        assert (output_dir / "js" / "list_distributor.js").exists()
        assert (output_dir / "data" / "config.json").exists()
        assert (output_dir / "data" / "lists.jsonl").exists()
        assert (output_dir / "data" / "items.jsonl").exists()
        assert (output_dir / "data" / "distribution.json").exists()

    def test_generate_with_sequential_strategy(
        self,
        runner: CliRunner,
        sample_lists_dir: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test generate command with sequential distribution strategy."""
        output_dir = tmp_path / "experiment"

        result = runner.invoke(
            generate,
            [
                str(sample_lists_dir),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type", "forced_choice",
                "--distribution-strategy", "sequential",
            ],
        )

        assert result.exit_code == 0

        # Verify distribution config
        dist_config = json.loads((output_dir / "data" / "distribution.json").read_text())
        assert dist_config["strategy_type"] == "sequential"

    def test_generate_with_quota_strategy(
        self,
        runner: CliRunner,
        sample_lists_dir: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test generate command with quota-based distribution strategy."""
        output_dir = tmp_path / "experiment"

        config_json = json.dumps({"participants_per_list": 10, "allow_overflow": False})

        result = runner.invoke(
            generate,
            [
                str(sample_lists_dir),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type", "likert_rating",
                "--distribution-strategy", "quota_based",
                "--distribution-config", config_json,
            ],
        )

        assert result.exit_code == 0

        # Verify distribution config
        dist_config = json.loads((output_dir / "data" / "distribution.json").read_text())
        assert dist_config["strategy_type"] == "quota_based"
        assert dist_config["strategy_config"]["participants_per_list"] == 10

    def test_generate_with_debug_mode(
        self,
        runner: CliRunner,
        sample_lists_dir: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test generate command with debug mode enabled."""
        output_dir = tmp_path / "experiment"

        result = runner.invoke(
            generate,
            [
                str(sample_lists_dir),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type", "forced_choice",
                "--distribution-strategy", "balanced",
                "--debug-mode",
                "--debug-list-index", "1",
            ],
        )

        assert result.exit_code == 0

        # Verify debug mode config
        dist_config = json.loads((output_dir / "data" / "distribution.json").read_text())
        assert dist_config["debug_mode"] is True
        assert dist_config["debug_list_index"] == 1

    def test_generate_missing_distribution_strategy(
        self,
        runner: CliRunner,
        sample_lists_dir: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test error when distribution strategy is missing."""
        output_dir = tmp_path / "experiment"

        result = runner.invoke(
            generate,
            [
                str(sample_lists_dir),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type", "likert_rating",
                # Missing --distribution-strategy (required)
            ],
        )

        # Should fail due to missing required option
        assert result.exit_code != 0

    def test_generate_invalid_distribution_config(
        self,
        runner: CliRunner,
        sample_lists_dir: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test error when distribution config JSON is invalid."""
        output_dir = tmp_path / "experiment"

        result = runner.invoke(
            generate,
            [
                str(sample_lists_dir),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type", "likert_rating",
                "--distribution-strategy", "quota_based",
                "--distribution-config", "not-valid-json",
            ],
        )

        # Should fail due to invalid JSON
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output


# ==================== Validate Command Tests ====================


class TestValidateCommand:
    """Test deployment validate command."""

    def test_validate_valid_experiment(
        self,
        runner: CliRunner,
        sample_lists_dir: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test validate command on valid experiment directory."""
        output_dir = tmp_path / "experiment"

        # First generate an experiment
        runner.invoke(
            generate,
            [
                str(sample_lists_dir),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type", "likert_rating",
                "--distribution-strategy", "balanced",
            ],
        )

        # Then validate it
        result = runner.invoke(
            validate,
            [str(output_dir)],
        )

        assert result.exit_code == 0, f"Validation failed: {result.output}"

    def test_validate_missing_directory(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test validate command on non-existent directory."""
        nonexistent_dir = tmp_path / "does_not_exist"

        result = runner.invoke(
            validate,
            [str(nonexistent_dir)],
        )

        # Should fail due to missing directory
        assert result.exit_code != 0

    def test_validate_incomplete_structure(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test validate command on incomplete experiment directory."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        # Create only partial structure
        (exp_dir / "index.html").write_text("<html></html>")

        result = runner.invoke(
            validate,
            [str(exp_dir)],
        )

        # Should fail due to missing required files
        assert result.exit_code != 0


# ==================== Export JATOS Command Tests ====================


class TestExportJATOSCommand:
    """Test deployment export-jatos command."""

    def test_export_jatos_basic(
        self,
        runner: CliRunner,
        sample_lists_dir: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test export-jatos command with basic options."""
        output_dir = tmp_path / "experiment"
        jzip_path = tmp_path / "study.jzip"

        # First generate an experiment
        runner.invoke(
            generate,
            [
                str(sample_lists_dir),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type", "likert_rating",
                "--distribution-strategy", "balanced",
            ],
        )

        # Then export to JATOS
        result = runner.invoke(
            export_jatos,
            [
                str(output_dir),
                str(jzip_path),
                "--title", "Test Study",
                "--description", "Test Description",
            ],
        )

        assert result.exit_code == 0
        assert jzip_path.exists()
        assert jzip_path.suffix == ".jzip"


# ==================== Distribution Strategy Integration Tests ====================


class TestDistributionStrategies:
    """Test all 8 distribution strategies."""

    @pytest.mark.parametrize("strategy", [
        "random",
        "sequential",
        "balanced",
        "latin_square",
    ])
    def test_simple_strategies(
        self,
        runner: CliRunner,
        sample_lists_dir: Path,
        sample_items_file: Path,
        tmp_path: Path,
        strategy: str,
    ) -> None:
        """Test simple distribution strategies (no config required)."""
        output_dir = tmp_path / f"experiment_{strategy}"

        result = runner.invoke(
            generate,
            [
                str(sample_lists_dir),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type", "forced_choice",
                "--distribution-strategy", strategy,
            ],
        )

        assert result.exit_code == 0, f"Strategy {strategy} failed: {result.output}"

        # Verify distribution config
        dist_config = json.loads((output_dir / "data" / "distribution.json").read_text())
        assert dist_config["strategy_type"] == strategy

    def test_stratified_strategy(
        self,
        runner: CliRunner,
        sample_lists_dir: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test stratified distribution strategy with factors."""
        output_dir = tmp_path / "experiment"

        config_json = json.dumps({"factors": ["condition"]})

        result = runner.invoke(
            generate,
            [
                str(sample_lists_dir),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type", "likert_rating",
                "--distribution-strategy", "stratified",
                "--distribution-config", config_json,
            ],
        )

        assert result.exit_code == 0

        # Verify distribution config
        dist_config = json.loads((output_dir / "data" / "distribution.json").read_text())
        assert dist_config["strategy_type"] == "stratified"
        assert dist_config["strategy_config"]["factors"] == ["condition"]

    def test_weighted_random_strategy(
        self,
        runner: CliRunner,
        sample_lists_dir: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test weighted_random distribution strategy."""
        output_dir = tmp_path / "experiment"

        config_json = json.dumps({"weight_expression": "1.0", "normalize_weights": True})

        result = runner.invoke(
            generate,
            [
                str(sample_lists_dir),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type", "likert_rating",
                "--distribution-strategy", "weighted_random",
                "--distribution-config", config_json,
            ],
        )

        assert result.exit_code == 0

    def test_metadata_based_strategy(
        self,
        runner: CliRunner,
        sample_lists_dir: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test metadata_based distribution strategy."""
        output_dir = tmp_path / "experiment"

        config_json = json.dumps({
            "filter_expression": "true",
            "rank_expression": "list_metadata.list_number || 0",
            "rank_ascending": True
        })

        result = runner.invoke(
            generate,
            [
                str(sample_lists_dir),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type", "likert_rating",
                "--distribution-strategy", "metadata_based",
                "--distribution-config", config_json,
            ],
        )

        assert result.exit_code == 0


# ==================== Help Command Tests ====================


class TestHelpCommands:
    """Test help output for deployment commands."""

    def test_deployment_help(self, runner: CliRunner) -> None:
        """Test deployment command help."""
        result = runner.invoke(deployment, ["--help"])

        assert result.exit_code == 0
        assert "Deployment commands" in result.output

    def test_generate_help(self, runner: CliRunner) -> None:
        """Test generate command help."""
        result = runner.invoke(deployment, ["generate", "--help"])

        assert result.exit_code == 0
        assert "Generate jsPsych experiment" in result.output or "generate" in result.output.lower()

    def test_export_jatos_help(self, runner: CliRunner) -> None:
        """Test export-jatos command help."""
        result = runner.invoke(deployment, ["export-jatos", "--help"])

        assert result.exit_code == 0
        assert "export" in result.output.lower() or "JATOS" in result.output
