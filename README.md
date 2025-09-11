```shell
git clone --recurse-submodules git@gitlab.alibaba-inc.com:EconML/BeyondAgent.git
```


# BeyondAgent

This example demonstrates how to perform agent training for a given environment, e.g. appworld.

### 这种实现方式：

1. 目录结构: 对verl目录下的代码不做任何改动，所有与beyondagent的代码存放在recipe/beyond_agent目录下。
2. Trainer继承: 继承verl中RayTrainer类，对部分函数进行修改
3. ParallelEnvManager类：在Trainer中引入ParallelEnvManager类，可通过线程池，并行执行多个dataflow对象（为每个prompt创建一个dataflow对象），并对输出结果进行聚合（upcoming）。
4. AsyncLLMServerManager类：在ParallelEnvManager中使用verl中的LLMServerManager类，所有dataflow对象共用同一个LLMServerManager，由LLMServerManager同时管理多个vLLM server, 通过ChatScheduler对来自各个线程中dataflow的llm-call进行分发和等待。



## Installation Guide

```bash
# use uv to install deps, you can also choose conda
uv venv --python=3.11
source .venv/bin/activate
# clone our verl branch
git submodule update --init external/verl
# make sure our pip is ready
uv pip install --upgrade pip setuptools packaging -i https://mirrors.aliyun.com/pypi/simple/
# install the majority of dependencies
uv pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --prerelease=allow
# create link to verl
uv pip install -e external/verl -i https://mirrors.aliyun.com/pypi/simple/
# finally, install flash attention (must be installed at last, need to connect to github)
uv pip install --verbose flash-attn==2.7.4.post1 ring-flash-attn -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --no-build-isolation
```


## Usage

### Step 1: Install & Run EnvService

```bash
cd envservice
python3 -m env.env_service
```

### Step 2: Run BeyondAgent Training

If you have 2 GPUs
Use the standard 2-GPU script:

```bash
cd your_verl_root_dir
bash examples/run_qwen2.5-3b_dataflow_2gpu.sh
```

## Launcher Usage

`Launcher` is a one-stop experiment manager that can start and backup experiments automatically.

1. Launching experiment from yaml, start environment service manually.

```
python launcher.py --conf examples/example_launcher/anni_baseline.yaml
```


2. Launching environment service with `Launcher` (using appworld as example)
    - edit `./.env`
    ```
    APPWORLD_PATH=...
    APPWORLD_ACTIVATION=...
    ```

    - run
    ```
    python launcher.py --with-appworld
    ```


3. Launching environment service **before** starting the training. (Automatically capture the success message of environment service)