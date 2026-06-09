# Installation Guide

## Requirements

- System: Linux with NVIDIA GPU
- Python 3.10
- NVIDIA Isaac Sim 4.5 + Isaac Lab 2.1.1
- [NVIDIA cuRobo](https://curobo.org)
- [TacEx](https://github.com/DH-Ng/TacEx): **Must be built from the local `third_party/TacEx` source** (contains project-specific modifications)

## Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/byml-c/UniVTAC.git
cd UniVTAC
```

### Prerequisites

Install system-level build dependencies (Ubuntu/Debian):

```bash
sudo apt install -y cmake build-essential gcc-11 g++-11 pkg-config
```

For CUDA Toolkit 12.4, install from [NVIDIA's download archive](https://developer.nvidia.com/cuda-12-4-0-download-archive), or use your system CUDA if already installed (check with `nvcc --version`).

> **Note:** `uv` manages Python packages, but system-level tools (gcc, cmake, CUDA) must be installed separately. The original conda environment file at `third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml` lists the required versions for reference.

### Step 2: Create a Virtual Environment with uv

```bash
uv venv --python 3.10 --seed
source .venv/bin/activate
```

### Step 3: Install TacEx (Modified Source)

> **Important:** Do **not** install TacEx from the public repository. UniVTAC requires a modified version of TacEx that is bundled in `third_party/TacEx`. Some internal APIs have been adapted for UniVTAC's tactile sensor pipeline.

``` bash
cd third_party/TacEx
```

If you have a working Isaac Lab environment, you can directly install TacEx. Otherwise, **you need to install Isaac Sim 4.5 and Isaac Lab 2.1.1**. Below is a quick summary, but here is the [full installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

<details>
<summary>Quick summary for Installing Isaac Sim and Isaac Lab for Ubuntu 22.04</summary>

> [!note]
> To install Isaac Sim for Ubuntu 20.04 follow the [binary installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html).

#### Isaac Sim - Linux pip installation

```bash
# install cuda-enabled pytorch
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
# install isaac sim packages
uv pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

> verify that the Isaac Sim installation works by calling `isaacsim` in the terminal

#### Isaac Lab

```bash
# install dependencies via apt (Ubuntu)
sudo apt install cmake build-essential
git clone https://github.com/isaac-sim/IsaacLab
cd IsaacLab
# use Isaac Lab version 2.1.1
git checkout v2.1.1
# activate the Isaac Sim python env
source .venv/bin/activate
# install isaaclab extensions (with --editable flag)
./isaaclab.sh --install # or "./isaaclab.sh -i"
```

To verify the Isaac Lab Installation:

```bash
source .venv/bin/activate
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
```

</details>

#### Installing TacEx [Core]

**1.** Activate the Isaac Env
```bash
source .venv/bin/activate
```

**2.** Install the core packages of TacEx
```bash
# Script will pip install core TacEx packages with --editable flag)
./tacex.sh -i
```

> You can install the extensions one by one via e.g. `uv pip install -e source/tacex_uipc`
>
> **Note:** `tacex`, `tacex_assets`, and `tacex_tasks` import Isaac Sim modules (`omni`, `isaaclab_tasks`) at runtime. Make sure Isaac Sim is installed before verifying the import.

**3.** Verify that TacEx works by running an example:

```bash
python ./scripts/demos/tactile_sim_approaches/check_taxim_sim.py --debug_vis
```

And here is an RL example:
```bash
python ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-RGB-v0 --num_envs 512 --enable_cameras
```
> You can view the sensor output in the IsaacLab Tab: `Scene Debug Visualization > Observations > sensor_output`

#### Installing TacEx [UIPC]
The `tacex_uipc` package is responsible for the [UIPC](https://spirimirror.github.io/libuipc-doc/) simulation in TacEx.

**1.** Install the [libuipc dependencies](https://spirimirror.github.io/libuipc-doc/build_install/linux/):
* If not installed yet, install Vcpkg

```bash
mkdir ~/Toolchain
cd ~/Toolchain
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh -disableMetrics
```

* Set the System Environment Variable  `CMAKE_TOOLCHAIN_FILE` to let CMake detect Vcpkg. If you installed it like above, you can do this:

```bash
# Write in ~/.bashrc
export CMAKE_TOOLCHAIN_FILE="$HOME/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

* We also need `CMake 3.26`, `GCC 11.4` and `Cuda 12.4` to build libuipc. Install this into the Isaac Sim python env:

```bash
# Inside the root dir of TacEx repo
source .venv/bin/activate
```
> **Note:** The build dependencies (`cmake 3.26`, `gcc 11.4`, `cuda-toolkit 12.4`) must be installed at the system level — see [Prerequisites](#prerequisites) above. The original conda environment file is at `source/tacex_uipc/libuipc/conda/env.yaml` for reference.
> If Cuda 12.4 does not work, try updating your Nvidia drivers or use an older Cuda version.

**2.** Install `tacex_uipc`
```bash
# This also builds `libuipc` and pip installs the python bindings.
source .venv/bin/activate
uv pip install -e source/tacex_uipc -v
```
> You can also install all TacEx packages with `./tacex.sh -i all`.

**3.** Verify that the `tacex_uipc` works by running an example:

```bash
python ./scripts/benchmarking/tactile_sim_performance/run_ball_rolling_experiment.py --num_envs 1 --debug_vis --env uipc
```

### Step 4: Install cuRobo

cuRobo is used for GPU-accelerated collision-aware motion planning. Follow the official [cuRobo Installation Guide](https://curobo.org/get_started/1_install_instructions.html).