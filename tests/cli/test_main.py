"""Tests for main CLI commands."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from sash import __version__
from sash.cli.main import cli


def test_cli_version(cli_runner: CliRunner) -> None:
    """Test --version option."""
    result = cli_runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_help(cli_runner: CliRunner) -> None:
    """Test --help option."""
    result = cli_runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "sash" in result.output
    assert "Semantic Acceptability" in result.output


def test_cli_with_config_file(cli_runner: CliRunner, mock_config_file: Path) -> None:
    """Test CLI with --config-file option."""
    result = cli_runner.invoke(cli, ["--config-file", str(mock_config_file)])
    assert result.exit_code == 0


def test_cli_with_profile(cli_runner: CliRunner) -> None:
    """Test CLI with --profile option."""
    result = cli_runner.invoke(cli, ["--profile", "dev"])
    assert result.exit_code == 0


def test_cli_with_verbose(cli_runner: CliRunner) -> None:
    """Test CLI with --verbose option."""
    result = cli_runner.invoke(cli, ["--verbose"])
    assert result.exit_code == 0


def test_cli_with_quiet(cli_runner: CliRunner) -> None:
    """Test CLI with --quiet option."""
    result = cli_runner.invoke(cli, ["--quiet"])
    assert result.exit_code == 0


def test_init_command_help(cli_runner: CliRunner) -> None:
    """Test init command help."""
    result = cli_runner.invoke(cli, ["init", "--help"])
    assert result.exit_code == 0
    assert "Initialize" in result.output or "init" in result.output.lower()


def test_init_command_creates_project(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test init command creates project structure."""
    project_dir = tmp_path / "my_project"

    result = cli_runner.invoke(
        cli,
        ["init", "my_project"],
        cwd=str(tmp_path),
        catch_exceptions=False,
    )

    # Check exit code
    assert result.exit_code == 0

    # Check project directory created
    assert project_dir.exists()

    # Check subdirectories
    expected_dirs = [
        "lexicons",
        "templates",
        "filled_templates",
        "items",
        "lists",
        "experiments",
        "data",
        "models",
    ]

    for subdir in expected_dirs:
        assert (project_dir / subdir).exists(), f"Missing directory: {subdir}"

    # Check files
    assert (project_dir / "sash.yaml").exists()
    assert (project_dir / ".gitignore").exists()

    # Check config content
    config_content = (project_dir / "sash.yaml").read_text()
    assert "profile:" in config_content
    assert "paths:" in config_content


def test_init_command_with_profile(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test init command with profile option."""
    result = cli_runner.invoke(
        cli,
        ["init", "test_project", "--profile", "dev"],
        cwd=str(tmp_path),
    )

    assert result.exit_code == 0
    project_dir = tmp_path / "test_project"
    config_content = (project_dir / "sash.yaml").read_text()
    assert "profile: dev" in config_content


def test_init_command_current_directory(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test init command in current directory."""
    result = cli_runner.invoke(cli, ["init"], cwd=str(tmp_path))

    assert result.exit_code == 0
    assert (tmp_path / "sash.yaml").exists()


def test_init_command_existing_directory_with_force(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    """Test init command with --force on existing directory."""
    project_dir = tmp_path / "existing_project"
    project_dir.mkdir()

    # Create existing file
    (project_dir / "existing.txt").write_text("existing")

    result = cli_runner.invoke(
        cli,
        ["init", "existing_project", "--force"],
        cwd=str(tmp_path),
    )

    assert result.exit_code == 0
    assert (project_dir / "sash.yaml").exists()


def test_init_command_invalid_project_name(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    """Test init command with invalid project name."""
    result = cli_runner.invoke(
        cli,
        ["init", "invalid name with spaces"],
        cwd=str(tmp_path),
    )

    assert result.exit_code == 0  # Command runs but shows error
    assert "Invalid project name" in result.output


def test_init_command_existing_nonempty_directory_no_force(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    """Test init command on existing non-empty directory without --force."""
    project_dir = tmp_path / "nonempty"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("content")

    # Simulate user declining confirmation
    result = cli_runner.invoke(
        cli,
        ["init", "nonempty"],
        cwd=str(tmp_path),
        input="n\n",  # Answer 'no' to confirmation
    )

    # Should exit gracefully
    assert result.exit_code == 0


def test_gitignore_content(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test .gitignore file content."""
    result = cli_runner.invoke(
        cli,
        ["init", "test_gitignore"],
        cwd=str(tmp_path),
    )

    assert result.exit_code == 0

    gitignore_path = tmp_path / "test_gitignore" / ".gitignore"
    gitignore_content = gitignore_path.read_text()

    # Check for important entries
    assert ".cache/" in gitignore_content
    assert "__pycache__/" in gitignore_content
    assert "*.pyc" in gitignore_content
    assert ".DS_Store" in gitignore_content


def test_config_yaml_structure(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test generated sash.yaml structure."""
    result = cli_runner.invoke(
        cli,
        ["init", "test_config"],
        cwd=str(tmp_path),
    )

    assert result.exit_code == 0

    config_path = tmp_path / "test_config" / "sash.yaml"
    config_content = config_path.read_text()

    # Check for all major sections
    sections = [
        "profile:",
        "logging:",
        "paths:",
        "resources:",
        "templates:",
        "models:",
        "items:",
        "lists:",
        "deployment:",
        "training:",
    ]

    for section in sections:
        assert section in config_content, f"Missing section: {section}"
