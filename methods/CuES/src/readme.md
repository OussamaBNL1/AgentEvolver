# src/ directory overview

This directory contains the core source code of AgentFlow, organized by responsibility. Below is a brief description of each subdirectory.

- agents/
  - Purpose: Agent logic and trajectory evaluation.
  - Key files:
    - llm_agent.py: LLM-based plan–act–observe loop for interacting with environments.
    - trajectory_evaluator.py: Quality checks and metrics over generated trajectories.
  - Edit when: You need to change decision-making or evaluation criteria.

- core/
  - Purpose: Orchestration and shared infrastructure.
  - Key files:
    - pipeline.py: Coordinator for Stage 1/2/3, supports full and per-stage runs, statistics, and optional Query Rewrite.
    - api_client.py: DashScope (or other providers) API wrapper with retries and timeouts.
    - memory_manager.py: Context/memory management utilities (if used).
  - Edit when: You need to adjust overall flow, swap model providers, or add cross-cutting features.

- data/
  - Purpose: Data models and persistence helpers.
  - Key files:
    - models.py: Definitions for Triplet, Task, Session, Trajectory.
    - storage.py: JSON/JSONL I/O and directory layout under data_dir.
  - I/O layout:
    - Stage 1: data/triplets/*.jsonl
    - Stage 2: data/tasks/*.jsonl
    - Stage 3: data/trajectories/trajectory_*.json and data/trajectories/failed_tasks/*.json
    - Query Rewrite: data/rewrites/trajectories/trajectory_*_rw*.json

- environment/
  - Purpose: Environment abstraction and simple factory.
  - Key files:
    - env_factory.py: Create environment instances based on config.environment.type.
    - base_env.py: Minimal environment interface (if present).
  - Edit when: You add lightweight/local environments or adapt different environment types.

- envs/
  - Purpose: EnvService-backed environment managers.
  - Typical files:
    - appworld_manager.py, bfcl_manager.py, webshop_manager.py: HTTP-based integration to external env services.
  - Edit when: You integrate or extend EnvService-driven environments.

- stages/
  - Purpose: Implementations of the three stages.
  - Key files:
    - stage1_triplet_generation.py: Explore environments to generate triplets (state, action, reward).
    - stage2_task_abstraction.py: Abstract executable tasks and queries from triplets.
    - stage3_trajectory_generation.py: Execute tasks and produce full trajectories with messages.
  - Edit when: You change algorithms, batching/concurrency, or output formats per stage.

- prompts/
  - Purpose: Prompt templates and parsers.
  - Content: Prompt builders for exploration, abstraction, evaluation, rewrite, etc.
  - Edit when: You tune prompts, add strategies, or adapt to different models.

- utils/
  - Purpose: Shared utilities.
  - Key files:
    - logger.py: Unified logging setup and formatting.
    - Misc helpers: paths, timing, validation, etc.

## Typical call chain

main.py → core/pipeline.py (AgentFlowPipeline) → stages (Stage 1/2/3) → agents/envs/data

Configuration entry: config/config.yaml (api, environment, stage1–3, threading, rewrite)

## Extension guide

- New environment (EnvService-backed):
  - Add {your_env}_manager.py in envs/ and wire it in environment/env_factory.py.
- New model/provider:
  - Implement a client in core/api_client.py and inject it via core/pipeline.py.
- New stage or strategy variants:
  - Add a new implementation in stages/ and register/orchestrate it in core/pipeline.py.
