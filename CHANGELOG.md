# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-26

### Added

- **6-stage pipeline** for linguistic experiment design:
  - Resources: lexical items and templates with constraints
  - Templates: filling strategies (exhaustive, random, stratified, MLM, mixed)
  - Items: experimental item construction (8 task types)
  - Lists: list partitioning with constraint satisfaction
  - Deployment: jsPsych 8.x batch experiment generation for JATOS
  - Training: active learning with GLMM support and convergence detection

- **8 task types**: forced-choice, ordinal scale, binary, categorical, multi-select, magnitude, free text, cloze

- **Stand-off annotation** with UUID-based references for provenance tracking

- **GLMM support**: Generalized Linear Mixed Models with random effects

- **Batch deployment**: server-side list distribution via JATOS batch sessions

- **Language-agnostic design**: works with any language supported by UniMorph

- **Configuration-first approach**: single YAML file orchestrates entire pipeline

- **Type-safe implementation**: full Python 3.13 type hints with Pydantic v2 validation

- **CLI commands**:
  - `bead init`: project scaffolding
  - `bead config`: configuration management
  - `bead resources`: lexicon and resource management
  - `bead templates`: template filling operations
  - `bead items`: item construction
  - `bead lists`: list partitioning
  - `bead deployment`: experiment export for JATOS
  - `bead simulate`: annotation simulation
  - `bead training`: model training with active learning
  - `bead workflow`: pipeline orchestration
  - `bead shell`: interactive REPL

- **TypeScript support** for jsPsych plugins with Biome linting and type checking

- **Documentation** with MkDocs and mkdocstrings

- **CI/CD workflows** for testing, documentation deployment, and PyPI publishing

[Unreleased]: https://github.com/caulking/bead/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/caulking/bead/releases/tag/v0.1.0
