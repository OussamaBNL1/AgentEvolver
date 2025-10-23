
## 🚀 Quick Start
### Step 1. Basic Dependency Installation

First, clone all submodule.
```bash
git submodule update --init external/verl
```

Then, set up the training environment, choose between `uv` and `conda`.

<details>
<summary>🛠️ Set up environment with uv (Click to read detail)</summary>

```bash
# 🧰 setup uv (you can also choose conda if you prefer, but conda is too slow)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python=3.11 # If this step is slow, add ENV variable: UV_PYTHON_INSTALL_MIRROR="https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download"
source .venv/bin/activate
# 🌱 clone our verl branch
git submodule update --init external/verl
# 🆙 make sure our pip is ready
uv pip install --upgrade pip setuptools packaging -i https://mirrors.aliyun.com/pypi/simple/
uv pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --prerelease=allow
uv pip install -e external/verl -i https://mirrors.aliyun.com/pypi/simple/
# ✨ finally, install flash attention (must be installed at last, need to connect to github)
uv pip install --verbose flash-attn==2.7.4.post1 ring-flash-attn -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --no-build-isolation
```

</details>
<details>
<summary>🛠️ Set up environment with conda (Click to read detail)</summary>

```bash
conda create -n appworld python=3.11 -y
conda activate appworld
# 🆙 make sure our pip is ready
pip install --upgrade pip setuptools packaging -i https://mirrors.aliyun.com/pypi/simple/
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --prerelease=allow
pip install -e external/verl -i https://mirrors.aliyun.com/pypi/simple/
pip install --verbose flash-attn==2.7.4.post1 ring-flash-attn -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --no-build-isolation
```

</details>

### Step 2. Setup Env-Service (Appworld as example)
The script below sets up an environment for appworld. For other environment setup, refer to [docs/guidelines/env_service.md](docs/guidelines/env_service.md) 📄

```bash
cd env_service/environments/appworld && bash setup.sh
```

### Step 3. Begin Training! 🚀 🚀

```bash
python launcher.py --conf examples/self-question-attr.yaml --with-appworld --with-logview
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

```bash
python launcher.py --conf examples/example_launcher/anni_baseline.yaml
```


2. Launching environment service with `Launcher` (using appworld as example)
    - edit `./.env`, or use `export`
    ```bash
    APPWORLD_PATH=...
    APPWORLD_ACTIVATION=...
    ```

    - run
    ```bash
    python launcher.py --with-appworld
    ```


3. Launching environment service **before** starting the training. (Automatically capture the success message of environment service)

```bash
python launcher.py --conf examples/example_launcher/anni_baseline.yaml --with-appworld
```