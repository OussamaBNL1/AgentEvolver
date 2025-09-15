# from best_logger import print_dict
import subprocess
import argparse
import shutil
import time
import sys
import os
from dotenv import load_dotenv

load_dotenv()

BACK_TARGETS = os.environ.get('BACK_TARGETS', './config,./beyondagent,./context_manager_templates').split(',')


def parse_args():
    parser = argparse.ArgumentParser(description='BA Launcher')
    parser.add_argument(
        '--target',
        type=str,
        default='beyondagent.main_ppo',
        required=False,
        help='Target script to run (default: beyondagent.main_ppo)'
    )
    parser.add_argument('--conf',
        type=str,
        default="",
        required=False,
        help='Path to configuration file'
    )
    parser.add_argument('--db',
        type=str,
        default="",
        required=False,
        help='Path to configuration file'
    )
    parser.add_argument('--with-appworld',
        action='store_true',
        default=False,
        help='Launch appworld'
    )
    parser.add_argument('--with-webshop',
        action='store_true',
        default=False,
        help='Launch webshop'
    )
    parser.add_argument('--with-logview',
        action='store_true',
        default=False,
        help='Launch logview'
    )
    parser.add_argument('--with-crafters',
        action='store_true',
        default=False,
        help='Launch Crafters Env Simulation'
    )
    parser.add_argument('--reboot',
        action='store_true',
        default=False,
        help='reboot flag'
    )

    return parser.parse_args()

def main():
    args = parse_args()

    if args.conf:
        yaml_path = args.conf
        assert yaml_path.endswith('.yaml'), "Configuration file must be a YAML file"
        exp_base = os.path.dirname(args.conf)

        if os.path.exists(exp_base):

            ## 0. read yaml (get trainer.experiment_name)
            import yaml
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            exp_name = config.get('trainer').get('experiment_name')
            exp_name = exp_name.replace('|', '-')

            ## 1. check exp_base/backup exist
            backup_dir = os.path.join(exp_base, exp_name, 'backup')
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            else:
                total_seconds = 10
                for i in range(total_seconds):
                    print(f"\rWarning: backup directory already exists, we will automatically ignore this after {total_seconds - i} seconds...", end="", flush=True)
                    time.sleep(1)

            ## 2. copy files to backup
            for backup_target in BACK_TARGETS:
                print(f"Copying {backup_target} to {os.path.join(backup_dir, os.path.basename(backup_target))}")
                shutil.copytree(backup_target, os.path.join(backup_dir, os.path.basename(backup_target)), dirs_exist_ok=True)

            ## 3. copy yaml to backup
            yaml_backup_src = yaml_path
            yaml_backup_dst = os.path.join(exp_base, exp_name, 'yaml_backup.yaml')
            shutil.copyfile(yaml_backup_src, yaml_backup_dst)

        else:
            raise FileNotFoundError(f"Configuration file not found: {exp_base}")

        env = os.environ.copy()
        if args.db:
            env["RAY_DEBUG_POST_MORTEM"] = "1"
            env["DEBUG_TAGS"] = args.db
            env["RAY_record_task_actor_creation_sites"] =  "true"
            print("Debug mode is ON")
        else:
            print("Debug mode is OFF")
    else:
        assert args.with_appworld or args.with_webshop or args.with_logview or args.with_crafters, "You must at least do something."

    if args.with_appworld:
        from beyondagent.utils.daemon import LaunchCommandWhenAbsent
        appworld_path = os.environ.get('APPWORLD_PATH')
        appworld_activation = os.environ.get('APPWORLD_ACTIVATION')
        if appworld_path and os.path.exists(appworld_path):
            companion = LaunchCommandWhenAbsent(
                full_argument_list=[appworld_activation],
                dir=appworld_path,
                tag="appworld_env_service",
                use_pty=True
            )
            companion.launch(
                launch_wait_time=1800,
                success_std_string="Starting server on",
            )
        else:
            raise RuntimeError("EnvService not found")


    if args.with_crafters:
        from beyondagent.utils.daemon import LaunchCommandWhenAbsent
        crafters_path = os.environ.get('CRAFTERS_PATH')
        crafters_activation = os.environ.get('CRAFTERS_ACTIVATION')
        if crafters_path and os.path.exists(crafters_path):
            companion = LaunchCommandWhenAbsent(
                full_argument_list=[crafters_activation],
                dir=crafters_path,
                tag="crafters_env_service",
                use_pty=True
            )
            companion.launch(
                launch_wait_time=1800,
                success_std_string="Starting server on",
            )
        else:
            raise RuntimeError("EnvService not found")


    if args.with_webshop:
        from beyondagent.utils.daemon import LaunchCommandWhenAbsent
        webshop_path = os.environ.get('WEBSHOP_PATH')
        webshop_python = os.environ.get('WEBSHOP_PYTHON')
        webshop_port = os.environ.get('WEBSHOP_PORT', '1907')
        webshop_env_port = os.environ.get('WEBSHOP_ENV_PORT', '8080')
        java_home = os.environ.get('JAVA_HOME')
        java_ld_library_path = os.environ.get('JAVA_LD_LIBRARY_PATH')
        search_engine_path = os.environ.get('SEARCH_ENGINE_PATH')
        webshop_root = os.environ.get('WEBSHOP_ROOT')
        items_attr_path = os.environ.get('ITEMS_ATTR_PATH')
        items_file_path = os.environ.get('ITEMS_FILE_PATH')
        pythonpath = os.environ.get('PYTHONPATH')
        if webshop_path and os.path.exists(webshop_path):
            companion = LaunchCommandWhenAbsent(
                full_argument_list=[
                    webshop_python,
                    '-m',
                    'env_sandbox.environments.webshop.SimServer_launch',
                    "--portal",
                    "127.0.0.1",
                    "--port",
                    webshop_port,
                ],
                dir=webshop_path,
                tag="webshop_sim_server"
            )

            companion.launch(launch_wait_time=1800, success_std_string="Uvicorn running on", env_dict={
                "JAVA_HOME": java_home,
                "JAVA_LD_LIBRARY_PATH": java_ld_library_path,
                "search_engine_path": search_engine_path,
                "webshop_root": webshop_root,
                "ITEMS_ATTR_PATH": items_attr_path,
                "ITEMS_FILE_PATH": items_file_path,
                "PYTHONPATH": pythonpath
            }, force_restart=args.reboot)

            companion = LaunchCommandWhenAbsent(
                full_argument_list=[
                    webshop_python,
                    '-m',
                    'env_sandbox.env_service',
                    "--env",
                    "webshop",
                    "--portal",
                    "127.0.0.1",
                    "--port",
                    webshop_env_port,
                ],
                dir=webshop_path,
                tag="webshop_env_service"
            )
            companion.launch(launch_wait_time=1800,success_std_string="Uvicorn running on", env_dict={
                "JAVA_HOME": java_home,
                "JAVA_LD_LIBRARY_PATH": java_ld_library_path
            }, force_restart=args.reboot)
        else:
            raise RuntimeError("EnvService not found")




    if args.with_logview:
        from beyondagent.utils.daemon import LaunchCommandWhenAbsent
        logview_nvm_dir = os.environ.get('LOGVIEW_NVM_DIR')
        logview_nvm_bin = os.environ.get('LOGVIEW_NVM_BIN')
        logview_path = os.environ.get('LOGVIEW_PATH')
        companion = LaunchCommandWhenAbsent(
            full_argument_list=[
                sys.executable,
                '-m',
                'web_display.go',
            ],
            dir='./',
            tag="logview"
        )
        companion.launch(launch_wait_time=1800, success_std_string="Server running on", env_dict={
            'NVM_DIR': logview_nvm_dir,
            'NVM_BIN': logview_nvm_bin,
            'PATH': logview_path + os.environ.get('PATH', '')
        })


    if args.conf:
        # let's begin the training process
        cmd = [
            sys.executable,
            '-m',
            args.target,
            '--config-path',
            os.path.abspath(exp_base),
            '--config-name',
            os.path.basename(yaml_path),
        ]

        if args.with_logview:
            env.update({
                'BEST_LOGGER_WEB_SERVICE_URL': os.environ.get('BEST_LOGGER_WEB_SERVICE_URL', 'http://127.0.0.1:8181/')
            })

        try:
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=os.path.abspath('./'), env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running subprocess: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()