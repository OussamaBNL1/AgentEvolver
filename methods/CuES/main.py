#!/usr/bin/env python3
"""
AgentFlow Main Program
Lightweight two-stage data generation: triplet generation + task abstraction
"""
import os
import sys
import argparse
import yaml
import traceback
from pathlib import Path
from typing import Dict, Any
import threading

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.pipeline import AgentFlowPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        raise


def check_envservice_for_appworld(config):
    """Probe EnvService if an EnvService-backed environment is selected (appworld/bfcl/webshop)."""
    env_type = config.get('environment', {}).get('type')
    if env_type in {'appworld', 'bfcl', 'webshop'}:
        logger.info(f"Detected '{env_type}' environment, checking EnvService status...")

        import requests
        server_url = config.get('environment', {}).get('envservice', {}).get('server_url', 'http://localhost:8080')

        try:
            # A simple GET to the base URL is enough for a quick liveness probe
            requests.get(server_url, timeout=2)
            logger.info(f"✅ EnvService is reachable: {server_url}")
            return True
        except Exception:
            logger.warning("⚠️  EnvService not reachable. Please start it and ensure clients use an IP, not 0.0.0.0 or localhost across machines.")
            logger.warning(f"   cd {Path(__file__).parent / 'EnvService'}")
            logger.warning("   python -m env.env_service")
            logger.warning("   or run: python start_envservice.py")
            return False

    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AgentFlow: Lightweight triplet data generation & task abstraction")
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "stage1", "stage2", "stage3"],
        default="all",
        help="Run stage: all, stage1 (triplet generation), stage2 (task abstraction), stage3 (trajectory generation)"
    )
    parser.add_argument(
        "--session-name",
        type=str,
        help="Session name"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Input data file (for stage2/stage3 mode)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--requirement",
        type=str,
        help="Specific exploration requirement (e.g., 'only explore Spotify functionality')"
    )
    parser.add_argument(
        "--extract",
        type=bool,
        default=False,
        help="Extracting concept sets from the original environment"
    )
    parser.add_argument(
        "--rewrite",
        "--query-rewrite",
        dest="rewrite",
        action="store_true",
        default=False,
        help="Enable Query Rewrite after Stage 3 to diversify queries"
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if not check_envservice_for_appworld(config):
        logger.error("❌ EnvService not running, cannot use AppWorld environment")
        return 1
    
    logger.info(f"Loaded configuration file: {args.config}")
    
    api_key = config.get('api', {}).get('dashscope_api_key')
    if not api_key:
        logger.error("Please set DashScope API key in configuration file")
        logger.error("You can get API key at: https://dashscope.console.aliyun.com/")
        return 1
    
    data_dir = config.get('data_dir', './data')
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Threading configuration: max_workers={config['threading']['max_workers']}, enabled={config['threading']['enabled']}")

    pipeline = AgentFlowPipeline(config)
    
    logger.info("=" * 60)
    logger.info("AgentFlow Lightweight Data Generation Pipeline")
    logger.info("=" * 60)
    concepts = None
    
    if args.stage == "all":
        logger.info("Running full three-stage pipeline...")
        if args.requirement:
            print(f"Exploration requirement: {args.requirement}")
        
        if args.extract:
            print("=== Stage 0: Requirement Confirm ===")
            concepts = pipeline.extract_concept(query_count = 100, max_workers=config['threading']['max_workers'], batch_size = 10)
            print(f"Extracting concept sets: {concepts}")
            result = pipeline.run_full_pipeline(session_name=args.session_name, requirement=args.requirement, concepts=concepts)
        else:
            result = pipeline.run_full_pipeline(session_name=args.session_name, requirement=args.requirement,)
        
        if result['success']:
            logger.info("🎉 Pipeline executed successfully!")
            logger.info(f"Session ID: {result['session_id']}")
            logger.info(f"Generated triplets: {result['triplets_count']}")
            logger.info(f"Abstracted tasks: {result['tasks_count']}")
            logger.info(f"Generated trajectories: {result.get('trajectories_count', 0)}")
            
            stats = result.get('statistics', {})
            if stats:
                triplet_stats = stats.get('triplets', {})
                task_stats = stats.get('tasks', {})
                trajectory_stats = stats.get('trajectories', {})
                
                logger.info("\n📊 Statistics:")
                logger.info(f"  Average triplet reward: {triplet_stats.get('avg_reward', 0):.3f}")
                logger.info(f"  Task success rate: {triplet_stats.get('success_rate', 0):.3f}")
                logger.info(f"  Average task confidence: {task_stats.get('avg_confidence', 0):.3f}")
                logger.info(f"  High confidence tasks: {task_stats.get('high_confidence_tasks', 0)}")
                logger.info(f"  Trajectory success rate: {trajectory_stats.get('success_rate', 0):.3f}")
                logger.info(f"  Average trajectory length: {trajectory_stats.get('avg_length', 0):.1f}")
            
            if args.rewrite:
                print("=== Stage 4: Query Rewrite ===")
                rewrite_cfg = config.get('rewrite', {})
                batch_size = rewrite_cfg.get('batch_size', 20)
                num_variants = rewrite_cfg.get('num_variants', 3)
                # Find the latest file from the default directory
                trajectories_file = Path('./data/trajectories')
                if not trajectories_file:
                    traj_dir = Path('./data/trajectories')
                    if traj_dir.exists():
                        traj_files = list(traj_dir.glob('*.jsonl'))
                        if traj_files:
                            trajectories_file = str(max(traj_files, key=lambda f: f.stat().st_mtime))
                            logger.info(f"Using latest trajectories file for rewrite: {trajectories_file}")
                        else:
                            logger.error("No trajectories files found in ./data/trajectories/")
                            return 1
                    else:
                        logger.error("Trajectories directory ./data/trajectories/ not found")
                        return 1
                rewrite_result = pipeline.run_query_rewrite(
                    trajectories_file=trajectories_file,
                    batch_size=batch_size,
                    num_variants=num_variants,
                    session_name=args.session_name
                )
                if rewrite_result.get('success'):
                    logger.info("📝 Query Rewrite finished.")
                    print(f"Rewritten samples: {rewrite_result.get('rewritten_count', 0)}")
                    logger.info(f"Rewrite output dir: {rewrite_result.get('output_dir')}")
                else:
                    logger.error(f"❌ Query Rewrite failed: {rewrite_result.get('error')}")

        else:
            logger.error(f"❌ Pipeline execution failed: {result['error']}")
            return 1
            
    elif args.stage == "stage1":
        logger.info("Running Stage 1: Triplet generation...")
        if args.requirement:
            logger.info(f"Exploration requirement: {args.requirement}")
        result = pipeline.run_stage1_only(requirement=args.requirement)
        
        if result['success']:
            logger.info("🎉 Stage 1 executed successfully!")
            logger.info(f"Session ID: {result['session_id']}")
            logger.info(f"Generated triplets: {result['triplets_count']}")
        else:
            logger.error(f"❌ Stage 1 execution failed: {result['error']}")
            return 1
            
    elif args.stage == "stage2":
        if not args.input_file:
            # Find the latest triplets file if no input file specified
            triplets_dir = Path('./data/triplets')
            if triplets_dir.exists():
                triplet_files = list(triplets_dir.glob('*.jsonl'))
                if triplet_files:
                    latest_file = max(triplet_files, key=lambda f: f.stat().st_mtime)
                    args.input_file = str(latest_file)
                    logger.info(f"Using latest triplets file: {args.input_file}")
                else:
                    logger.error("No triplets files found in ./data/triplets/")
                    return 1
            else:
                logger.error("Triplets directory ./data/triplets/ not found")
                return 1

            
        logger.info("Running Stage 2: Task abstraction...")
        result = pipeline.run_stage2_only(triplets_file=args.input_file)
        
        if result['success']:
            logger.info("🎉 Stage 2 executed successfully!")
            logger.info(f"Session ID: {result['session_id']}")
            logger.info(f"Abstracted tasks: {result['tasks_count']}")
        else:
            logger.error(f"❌ Stage 2 execution failed: {result['error']}")
            return 1
            
    elif args.stage == "stage3":
        if not args.input_file:
            # Find the latest triplets file if no input file specified
            tasks_dir = Path('./data/tasks')
            if tasks_dir.exists():
                triplet_files = list(tasks_dir.glob('*.jsonl'))
                if triplet_files:
                    latest_file = max(triplet_files, key=lambda f: f.stat().st_mtime)
                    args.input_file = str(latest_file)
                    logger.info(f"Using latest triplets file: {args.input_file}")
                else:
                    logger.error("No triplets files found in ./data/triplets/")
                    return 1
            else:
                logger.error("Triplets directory ./data/triplets/ not found")
                return 1

        logger.info("Running Stage 3: Trajectory generation...")
        result = pipeline.run_stage3_only(tasks_file=args.input_file)
        
        if result['success']:
            logger.info("🎉 Stage 3 executed successfully!")
            logger.info(f"Session ID: {result['session_id']}")
            logger.info(f"Generated trajectories: {result['trajectories_count']}")
            
            stats = result.get('statistics', {})
            if stats:
                trajectory_stats = stats.get('trajectories', {})
                logger.info("\n📊 Trajectory Statistics:")
                logger.info(f"  Success rate: {trajectory_stats.get('success_rate', 0):.3f}")
                logger.info(f"  Average length: {trajectory_stats.get('avg_length', 0):.1f}")
                logger.info(f"  Failed tasks: {trajectory_stats.get('failed_count', 0)}")
        else:
            logger.error(f"❌ Stage 3 execution failed: {result['error']}")
            return 1
    
    logger.info("✅ Execution completed!")
    return 0
        


def run_quick_demo():
    """Run quick demo"""
    print("=" * 60)
    print("AgentFlow Demo")
    print("=" * 60)
    
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print("❌ Configuration file not found, please create config/config.yaml")
        return
    

    config = load_config(config_path)
    api_key = config.get('api', {}).get('dashscope_api_key')
    # Sensitive information is blanked; please set the API key via environment variables or secure config file
    
    if not api_key:
        print("❌ Please set DashScope API key in configuration file")
        print("   1. Visit https://dashscope.console.aliyun.com/")
        print("   2. Get API key")
        print("   3. Edit dashscope_api_key in config/config.yaml (left empty; please fill manually)")
        print("   4. Re-run the program")
        return
    
    print("🚀 Starting AgentFlow demo...")
    
    # Add threading configuration for demo
    demo_config = config.copy()
    demo_config['stage1'] = {'rollout_num': 10, 'max_steps': 20}
    demo_config['stage2'] = {'batch_size': 20, 'min_confidence': 0.7}
    demo_config['stage3'] = {'max_trajectories': 1, 'max_steps_per_trajectory': 10}
    demo_config['threading'] = {
        'max_workers': 10,  # Use worker threads for demo
        'enabled': True
    }
    
    print(f"Demo using {demo_config['threading']['max_workers']} worker threads")
    
    pipeline = AgentFlowPipeline(demo_config)
    result = pipeline.run_full_pipeline(session_name="demo_session")
    
    if result['success']:
        print("✅ Demo executed successfully!")
        print(f"   Generated triplets: {result['triplets_count']}")
        print(f"   Abstracted tasks: {result['tasks_count']}")
        print(f"   Generated trajectories: {result.get('trajectories_count', 0)}")
        print(f"   Session ID: {result['session_id']}")
    else:
        print(f"❌ Demo execution failed: {result['error']}")


if __name__ == "__main__":
    # run_quick_demo()
    main()

