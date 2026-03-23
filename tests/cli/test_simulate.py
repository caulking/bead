"""Tests for simulation CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from click.testing import CliRunner

from bead.cli.simulate import simulate
from bead.items.item import Item


class TestSimulateRunCommand:
    """Tests for simulate run command."""

    def test_run_basic(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test basic simulation run."""
        # Create items
        template_id = uuid4()
        items_file = tmp_path / "items.jsonl"
        test_items = [
            Item(
                id=uuid4(),
                item_template_id=template_id,
                rendered_elements={"option_a": "A", "option_b": "B"},
                item_metadata={"n_options": 2},
            )
            for _ in range(5)
        ]
        with open(items_file, "w") as f:
            for item in test_items:
                f.write(item.model_dump_json() + "\n")

        output_file = tmp_path / "results.jsonl"

        # Mock SimulationRunner
        with patch("bead.cli.simulate.SimulationRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_results = {
                "item_ids": [str(item.id) for item in test_items],
                "annotator_0": [0] * 5,
                "annotator_1": [1] * 5,
                "annotator_2": [0] * 5,
            }
            mock_runner.run.return_value = mock_results
            mock_runner_class.return_value = mock_runner

            result = cli_runner.invoke(
                simulate,
                [
                    "run",
                    "--items",
                    str(items_file),
                    "--annotator",
                    "lm_score",
                    "--n-annotators",
                    "3",
                    "--output",
                    str(output_file),
                ],
            )

        assert result.exit_code == 0
        assert "Simulation complete" in result.output
        assert output_file.exists()

        # Verify output format
        with open(output_file) as f:
            records = [json.loads(line) for line in f if line.strip()]

        assert len(records) == 15  # 5 items × 3 annotators
        assert all("item_id" in r for r in records)
        assert all("annotator_id" in r for r in records)
        assert all("annotation" in r for r in records)

    def test_run_with_config(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test simulation run with config file."""
        template_id = uuid4()
        items_file = tmp_path / "items.jsonl"
        test_items = [
            Item(
                id=uuid4(),
                item_template_id=template_id,
                rendered_elements={"text": "test"},
                item_metadata={"scale_min": 1, "scale_max": 7},
            )
            for _ in range(3)
        ]
        with open(items_file, "w") as f:
            for item in test_items:
                f.write(item.model_dump_json() + "\n")

        # Create config file
        config_file = tmp_path / "config.json"
        config_data = {
            "annotator_configs": [
                {
                    "strategy": "lm_score",
                    "noise_model": {
                        "noise_type": "temperature",
                        "temperature": 2.0,
                    },
                    "random_state": 42,
                }
            ],
            "n_annotators": 2,
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        output_file = tmp_path / "results.jsonl"

        with patch("bead.cli.simulate.SimulationRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_results = {
                "item_ids": [str(item.id) for item in test_items],
                "annotator_0": [3, 4, 5],
                "annotator_1": [4, 5, 6],
            }
            mock_runner.run.return_value = mock_results
            mock_runner_class.return_value = mock_runner

            result = cli_runner.invoke(
                simulate,
                [
                    "run",
                    "--items",
                    str(items_file),
                    "--config",
                    str(config_file),
                    "--output",
                    str(output_file),
                ],
            )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_run_oracle_annotator(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test simulation with oracle annotator."""
        template_id = uuid4()
        items_file = tmp_path / "items.jsonl"
        test_items = [
            Item(
                id=uuid4(),
                item_template_id=template_id,
                rendered_elements={"text": "test"},
                item_metadata={},
            )
            for _ in range(4)
        ]
        with open(items_file, "w") as f:
            for item in test_items:
                f.write(item.model_dump_json() + "\n")

        output_file = tmp_path / "results.jsonl"

        with patch("bead.cli.simulate.SimulationRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_results = {
                "item_ids": [str(item.id) for item in test_items],
                "annotator_0": [0, 1, 0, 1],
            }
            mock_runner.run.return_value = mock_results
            mock_runner_class.return_value = mock_runner

            result = cli_runner.invoke(
                simulate,
                [
                    "run",
                    "--items",
                    str(items_file),
                    "--annotator",
                    "oracle",
                    "--n-annotators",
                    "1",
                    "--noise-type",
                    "none",
                    "--output",
                    str(output_file),
                ],
            )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_run_with_noise_models(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test simulation with different noise models."""
        template_id = uuid4()
        items_file = tmp_path / "items.jsonl"
        test_items = [
            Item(
                id=uuid4(),
                item_template_id=template_id,
                rendered_elements={"premise": "P", "hypothesis": "H"},
                item_metadata={"categories": ["A", "B", "C"]},
            )
            for _ in range(3)
        ]
        with open(items_file, "w") as f:
            for item in test_items:
                f.write(item.model_dump_json() + "\n")

        output_file = tmp_path / "results.jsonl"

        # Test with systematic noise
        with patch("bead.cli.simulate.SimulationRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_results = {
                "item_ids": [str(item.id) for item in test_items],
                "annotator_0": ["A", "B", "C"],
                "annotator_1": ["B", "C", "A"],
            }
            mock_runner.run.return_value = mock_results
            mock_runner_class.return_value = mock_runner

            result = cli_runner.invoke(
                simulate,
                [
                    "run",
                    "--items",
                    str(items_file),
                    "--annotator",
                    "random",
                    "--n-annotators",
                    "2",
                    "--noise-type",
                    "systematic",
                    "--bias-strength",
                    "0.3",
                    "--bias-type",
                    "length",
                    "--output",
                    str(output_file),
                ],
            )

        assert result.exit_code == 0

    def test_run_missing_items(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test run with missing items file."""
        output_file = tmp_path / "results.jsonl"

        result = cli_runner.invoke(
            simulate,
            [
                "run",
                "--items",
                str(tmp_path / "nonexistent.jsonl"),
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 2  # Click usage error
        assert "File not found" in result.output or "does not exist" in result.output

    def test_run_help(self, cli_runner: CliRunner) -> None:
        """Test run command help."""
        result = cli_runner.invoke(simulate, ["run", "--help"])

        assert result.exit_code == 0
        assert "Run multi-annotator simulation" in result.output


class TestSimulateConfigureCommand:
    """Tests for simulate configure command."""

    def test_configure_yaml(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test configuration creation with YAML format."""
        output_file = tmp_path / "config.yaml"

        result = cli_runner.invoke(
            simulate,
            [
                "configure",
                "--strategy",
                "lm_score",
                "--n-annotators",
                "5",
                "--noise-type",
                "temperature",
                "--temperature",
                "1.5",
                "--random-seed",
                "42",
                "--output",
                str(output_file),
                "--format",
                "yaml",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Configuration saved" in result.output

        # Verify YAML content (basic check)
        with open(output_file) as f:
            content = f.read()
        assert "lm_score" in content
        assert "temperature" in content

    def test_configure_json(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test configuration creation with JSON format."""
        output_file = tmp_path / "config.json"

        result = cli_runner.invoke(
            simulate,
            [
                "configure",
                "--strategy",
                "distance",
                "--n-annotators",
                "10",
                "--noise-type",
                "random",
                "--output",
                str(output_file),
                "--format",
                "json",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify JSON structure
        with open(output_file) as f:
            config = json.load(f)

        assert "annotator_configs" in config
        assert "n_annotators" in config
        assert config["n_annotators"] == 10

    def test_configure_with_bias(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test configuration with systematic bias."""
        output_file = tmp_path / "config.json"

        result = cli_runner.invoke(
            simulate,
            [
                "configure",
                "--strategy",
                "lm_score",
                "--noise-type",
                "systematic",
                "--bias-strength",
                "0.5",
                "--bias-type",
                "frequency",
                "--output",
                str(output_file),
                "--format",
                "json",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        with open(output_file) as f:
            config = json.load(f)

        noise_config = config["annotator_configs"][0]["noise_model"]
        assert noise_config["noise_type"] == "systematic"
        assert noise_config["bias_strength"] == 0.5
        assert noise_config["bias_type"] == "frequency"

    def test_configure_help(self, cli_runner: CliRunner) -> None:
        """Test configure command help."""
        result = cli_runner.invoke(simulate, ["configure", "--help"])

        assert result.exit_code == 0
        assert "Create simulation configuration" in result.output


class TestSimulateAnalyzeCommand:
    """Tests for simulate analyze command."""

    def test_analyze_basic(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test basic simulation analysis."""
        # Create mock simulation results
        results_file = tmp_path / "results.jsonl"

        results = []
        for i in range(10):  # 10 items
            for j in range(3):  # 3 annotators
                results.append(
                    {
                        "item_id": f"item_{i}",
                        "annotator_id": f"annotator_{j}",
                        "annotation": i % 3,  # Labels 0, 1, 2
                    }
                )

        with open(results_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        result = cli_runner.invoke(
            simulate,
            [
                "analyze",
                "--results",
                str(results_file),
            ],
        )

        assert result.exit_code == 0
        assert "Analysis Summary" in result.output
        assert "Total Items" in result.output
        assert "Total Annotators" in result.output

    def test_analyze_with_output(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test analysis with JSON output."""
        results_file = tmp_path / "results.jsonl"

        results = []
        for i in range(5):
            for j in range(2):
                results.append(
                    {
                        "item_id": f"item_{i}",
                        "annotator_id": f"annotator_{j}",
                        "annotation": 0 if j == 0 else 1,
                    }
                )

        with open(results_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        output_file = tmp_path / "analysis.json"

        # Mock agreement computation
        with patch("bead.cli.simulate.InterAnnotatorMetrics") as mock_metrics:
            mock_metrics.krippendorff_alpha.return_value = 0.75

            result = cli_runner.invoke(
                simulate,
                [
                    "analyze",
                    "--results",
                    str(results_file),
                    "--output",
                    str(output_file),
                ],
            )

        assert result.exit_code == 0
        assert output_file.exists()

        with open(output_file) as f:
            analysis = json.load(f)

        assert "n_items" in analysis
        assert "n_annotators" in analysis
        assert "total_annotations" in analysis

    def test_analyze_numeric_annotations(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test analysis with numeric annotations."""
        results_file = tmp_path / "results.jsonl"

        results = []
        for i in range(8):
            for j in range(4):
                results.append(
                    {
                        "item_id": f"item_{i}",
                        "annotator_id": f"annotator_{j}",
                        "annotation": (i * j) % 7 + 1,  # Numeric scale 1-7
                    }
                )

        with open(results_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        result = cli_runner.invoke(
            simulate,
            [
                "analyze",
                "--results",
                str(results_file),
            ],
        )

        assert result.exit_code == 0
        assert "Analysis Summary" in result.output

    def test_analyze_missing_file(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test analyze with missing results file."""
        result = cli_runner.invoke(
            simulate,
            [
                "analyze",
                "--results",
                str(tmp_path / "nonexistent.jsonl"),
            ],
        )

        assert result.exit_code == 1
        assert "File not found" in result.output or "does not exist" in result.output

    def test_analyze_help(self, cli_runner: CliRunner) -> None:
        """Test analyze command help."""
        result = cli_runner.invoke(simulate, ["analyze", "--help"])

        assert result.exit_code == 0
        assert "Analyze simulation results" in result.output


class TestListCommands:
    """Tests for list-annotators and list-noise-models commands."""

    def test_list_annotators(self, cli_runner: CliRunner) -> None:
        """Test list-annotators command."""
        result = cli_runner.invoke(simulate, ["list-annotators"])

        assert result.exit_code == 0
        assert "Available Annotator Types" in result.output
        assert "lm_score" in result.output
        assert "distance" in result.output
        assert "random" in result.output
        assert "oracle" in result.output

    def test_list_noise_models(self, cli_runner: CliRunner) -> None:
        """Test list-noise-models command."""
        result = cli_runner.invoke(simulate, ["list-noise-models"])

        assert result.exit_code == 0
        assert "Available Noise Models" in result.output
        assert "temperature" in result.output
        assert "systematic" in result.output
        assert "random" in result.output
        assert "none" in result.output


class TestSimulateHelp:
    """Tests for simulate command group help."""

    def test_simulate_help(self, cli_runner: CliRunner) -> None:
        """Test simulate command group help."""
        result = cli_runner.invoke(simulate, ["--help"])

        assert result.exit_code == 0
        assert "simulation" in result.output.lower()
        assert "run" in result.output
        assert "configure" in result.output
        assert "analyze" in result.output
        assert "list-annotators" in result.output
        assert "list-noise-models" in result.output
