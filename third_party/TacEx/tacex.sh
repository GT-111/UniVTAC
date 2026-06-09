#!/usr/bin/env bash

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set tab-spaces
tabs 4

# get source directory
#export ISAACLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# get source directory of TacEx
export TACEX_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#==
# Helper functions
#==

# check if running in docker
is_docker() {
    [ -f /.dockerenv ] || \
    grep -q docker /proc/1/cgroup || \
    [[ $(cat /proc/1/comm) == "containerd-shim" ]] || \
    grep -q docker /proc/mounts || \
    [[ "$(hostname)" == *"."* ]]
}

extract_isaacsim_path() {
    # Use the sym-link path to Isaac Sim directory
    local isaac_path=${ISAACLAB_PATH}/_isaac_sim
    # If above path is not available, try to find the path using python
    if [ ! -d "${isaac_path}" ]; then
        # Use the python executable to get the path
        local python_exe=$(extract_python_exe)
        # Retrieve the path importing isaac sim and getting the environment path
        if [ $(${python_exe} -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
            local isaac_path=$(${python_exe} -c "import isaacsim; import os; print(os.environ['ISAAC_PATH'])")
        fi
    fi
    # check if there is a path available
    if [ ! -d "${isaac_path}" ]; then
        # throw an error if no path is found
        echo -e "[ERROR] Unable to find the Isaac Sim directory: '${isaac_path}'" >&2
        echo -e "\tThis could be due to the following reasons:" >&2
        echo -e "\t1. Virtual environment is not activated." >&2
        echo -e "\t2. Isaac Sim pip package 'isaacsim-rl' is not installed." >&2
        echo -e "\t3. Isaac Sim directory is not available at the default path: ${ISAACLAB_PATH}/_isaac_sim" >&2
        # exit the script
        exit 1
    fi
    # return the result
    echo ${isaac_path}
}

# extract the python from isaacsim
extract_python_exe() {
    # check if using uv venv
    if ! [[ -z "${VIRTUAL_ENV}" ]]; then
        # use venv python
        local python_exe=${VIRTUAL_ENV}/bin/python
    # check if using conda
    elif ! [[ -z "${CONDA_PREFIX}" ]]; then
        # use conda python
        local python_exe=${CONDA_PREFIX}/bin/python
    else
        # use kit python
        local python_exe=${ISAACLAB_PATH}/_isaac_sim/python.sh

    if [ ! -f "${python_exe}" ]; then
            # note: we need to check system python for cases such as docker
            # inside docker, if user installed into system python, we need to use that
            # otherwise, use the python from the kit
            if [ $(python -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
                local python_exe=$(which python)
            fi
        fi
    fi
    # check if there is a python path available
    if [ ! -f "${python_exe}" ]; then
        echo -e "[ERROR] Unable to find any Python executable at path: '${python_exe}'" >&2
        echo -e "\tThis could be due to the following reasons:" >&2
        echo -e "\t1. Virtual environment is not activated." >&2
        echo -e "\t2. Isaac Sim pip package 'isaacsim-rl' is not installed." >&2
        echo -e "\t3. Python executable is not available at the default path: ${ISAACLAB_PATH}/_isaac_sim/python.sh" >&2
        exit 1
    fi
    # return the result
    echo ${python_exe}
}

# extract the simulator exe from isaacsim
extract_isaacsim_exe() {
    # obtain the isaac sim path
    local isaac_path=$(extract_isaacsim_path)
    # isaac sim executable to use
    local isaacsim_exe=${isaac_path}/isaac-sim.sh
    # check if there is a python path available
    if [ ! -f "${isaacsim_exe}" ]; then
        # check for installation using Isaac Sim pip
        # note: pip installed Isaac Sim can only come from a direct
        # python environment, so we can directly use 'python' here
        if [ $(python -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
            # Isaac Sim - Python packages entry point
            local isaacsim_exe="isaacsim isaacsim.exp.full"
        else
            echo "[ERROR] No Isaac Sim executable found at path: ${isaac_path}" >&2
            exit 1
        fi
    fi
    # return the result
    echo ${isaacsim_exe}
}

# auto-detect pip command: prefer uv pip if available, fall back to pip
_pip_install() {
    if command -v uv &> /dev/null; then
        uv pip install "$@"
    else
        python -m pip install "$@"
    fi
}

# check if input directory is a python extension and install the module
install_isaaclab_extension() {
    # if the directory contains setup.py then install the python module
    if [ -f "$1/setup.py" ]; then
        echo -e "\t module: $1"
        _pip_install --editable "$1" -v
    fi
}

# setup uv virtual environment for Isaac Lab
setup_uv_env() {
    # get environment name from input (optional, for naming the venv directory)
    local env_name=$1
    local env_dir="${TACEX_PATH}/.venv"

    # check uv is installed
    if ! command -v uv &> /dev/null
    then
        echo "[ERROR] uv could not be found. Install uv and try again:"
        echo "        curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi

    # check if the environment exists
    if [ -d "${env_dir}" ]; then
        echo -e "[INFO] uv environment already exists at '${env_dir}'."
    else
        echo -e "[INFO] Creating uv environment at '${env_dir}'..."
        uv venv --python 3.10 --seed "${env_dir}"
    fi

    # activate the environment
    source "${env_dir}/bin/activate"

    # set up environment variables
    export ISAACLAB_PATH="${ISAACLAB_PATH}"
    alias isaaclab="${ISAACLAB_PATH}/isaaclab.sh"
    export RESOURCE_NAME="IsaacSim"

    # source Isaac Sim environment variables if available
    local isaacsim_setup_script="${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh"
    if [ -f "${isaacsim_setup_script}" ]; then
        source "${isaacsim_setup_script}"
    fi

    # install some extra dependencies
    echo -e "[INFO] Installing extra dependencies (this might take a few minutes)..."
    uv pip install importlib_metadata &> /dev/null

    # add information to the user about alias
    echo -e "[INFO] Added 'isaaclab' alias for 'isaaclab.sh' script."
    echo -e "[INFO] Created uv environment at '${env_dir}'.\n"
    echo -e "\t\t1. To activate the environment, run:                source ${env_dir}/bin/activate"
    echo -e "\t\t2. To install TacEx extensions, run:                ./tacex.sh -i"
    echo -e "\t\t4. To perform formatting, run:                      ./tacex.sh -f"
    echo -e "\t\t5. To deactivate the environment, run:              deactivate"
    echo -e "\n"
}

# update the vscode settings from template and isaac sim settings
update_vscode_settings() {
    echo "[INFO] Setting up vscode settings..."
    # retrieve the python executable
    python_exe=$(extract_python_exe)
    # path to setup_vscode.py
    setup_vscode_script="${ISAACLAB_PATH}/.vscode/tools/setup_vscode.py"
    # check if the file exists before attempting to run it
    if [ -f "${setup_vscode_script}" ]; then
        ${python_exe} "${setup_vscode_script}"
    else
        echo "[WARNING] Unable to find the script 'setup_vscode.py'. Aborting vscode settings setup."
    fi
}

# print the usage description
print_help () {
    echo -e "\nusage: $(basename "$0") [-h] [-i] [-f] [-p] [-s] [-t] [-o] [-v] [-d] [-e] -- Utility to manage Isaac Lab."
    echo -e "\noptional arguments:"
    echo -e "\t-h, --help             Display the help content."
    echo -e "\t-i, --install [all]    Install the TacEx core packages. Use 'all' to also install all extra packages [tacex_uipc]."
    echo -e "\t-f, --format           Run pre-commit to format the code and check lints."
    echo -e "\t-p, --python           Run the python executable provided by Isaac Sim or virtual environment (if active)."
    echo -e "\t-s, --sim              Run the simulator executable (isaac-sim.sh) provided by Isaac Sim."
    echo -e "\t-t, --test             Run all python unittest tests."
    echo -e "\t-o, --docker           Run the docker container helper script (docker/container.sh)."
    echo -e "\t-v, --vscode           Generate the VSCode settings file from template."
    echo -e "\t-d, --docs             Build the documentation from source using sphinx."
    echo -e "\t-e, --venv [NAME]      Create a uv virtual environment for Isaac Lab. Default name is 'env_isaaclab'."
    echo -e "\n" >&2
}


#==
# Main
#==

# check argument provided
if [ -z "$*" ]; then
    echo "[Error] No arguments provided." >&2;
    print_help
    exit 1
fi

# pass the arguments
while [[ $# -gt 0 ]]; do
    # read the key
    case "$1" in
        -i|--install)
            # install the python packages in tacex/source directory
            echo "[INFO] Installing extensions inside the TacEx repository..."
            # recursively look into directories and install them
            # this does not check dependencies between extensions
            export -f extract_python_exe
            export -f install_isaaclab_extension
            export -f _pip_install
            # source directory
            # find -L "${TACEX_PATH}/source" -mindepth 1 -maxdepth 1 -type d -exec bash -c 'install_isaaclab_extension "{}"' \;
            # install core packages
            echo "[INFO] Installing package [tacex]..."
            _pip_install -e ${TACEX_PATH}/source/tacex

            echo "[INFO] Installing package [tacex_assets]..."
            _pip_install -e ${TACEX_PATH}/source/tacex_assets

            echo "[INFO] Installing package [tacex_tasks]..."
            _pip_install -e ${TACEX_PATH}/source/tacex_tasks

            if [ -z "$2" ]; then
                echo "[INFO] No extra packages installed."
            elif [ "$2" = "all" ]; then
                echo "[INFO] Installing package tacex_uipc..."
                _pip_install -e ${TACEX_PATH}/source/tacex_uipc -v
                # consume the extra argument so it isn't processed by the outer loop
                shift
            # else
            #     echo "[INFO] Installing rl-framework: $2"
            #     extension_name=$2
            #     shift # past argument
            fi

            # check if we are inside a docker container or are building a docker image
            # in that case don't setup VSCode since it asks for EULA agreement which triggers user interaction
            # if is_docker; then
            #     echo "[INFO] Running inside a docker container. Skipping VSCode settings setup."
            #     echo "[INFO] To setup VSCode settings, run 'isaaclab -v'."
            # else
            #     # update the vscode settings
            #     update_vscode_settings
            # fi

            # unset local variables
            unset extract_python_exe
            unset install_isaaclab_extension
            unset _pip_install
            shift # past argument
            ;;
        -e|--venv)
            # use default name if not provided
            if [ -z "$2" ]; then
                echo "[INFO] Using default environment name: env_isaaclab"
                env_name="env_isaaclab"
            else
                echo "[INFO] Using environment name: $2"
                env_name=$2
                shift # past argument
            fi
            # setup the uv environment for Isaac Lab
            setup_uv_env ${env_name}
            shift # past argument
            ;;
        -f|--format)
            # reset the python path to avoid conflicts with pre-commit
            # this is needed because the pre-commit hooks are installed in a separate virtual environment
            # and it uses the system python to run the hooks
            if [ -n "${VIRTUAL_ENV}" ] || [ -n "${CONDA_DEFAULT_ENV}" ]; then
                cache_pythonpath=${PYTHONPATH}
                export PYTHONPATH=""
            fi
            # run the formatter over the repository
            # check if pre-commit is installed
            if ! command -v pre-commit &>/dev/null; then
                echo "[INFO] Installing pre-commit..."
                pip install pre-commit
            fi
            # always execute inside the TacEx directory
            echo "[INFO] Formatting the repository..."
            cd ${TACEX_PATH}
            pre-commit run --all-files
            cd - > /dev/null
            # set the python path back to the original value
            if [ -n "${VIRTUAL_ENV}" ] || [ -n "${CONDA_DEFAULT_ENV}" ]; then
                export PYTHONPATH=${cache_pythonpath}
            fi
            shift # past argument
            # exit neatly
            break
            ;;
        -p|--python)
            # run the python provided by isaacsim
            python_exe=$(extract_python_exe)
            echo "[INFO] Using python from: ${python_exe}"
            shift # past argument
            ${python_exe} "$@"
            # exit neatly
            break
            ;;
        -s|--sim)
            # run the simulator exe provided by isaacsim
            isaacsim_exe=$(extract_isaacsim_exe)
            echo "[INFO] Running isaac-sim from: ${isaacsim_exe}"
            shift # past argument
            ${isaacsim_exe} --ext-folder ${TACEX_PATH}/source $@
            # exit neatly
            break
            ;;
        -t|--test)
            # run the python provided by isaacsim
            python_exe=$(extract_python_exe)
            shift # past argument
            ${python_exe} -m pytest ${TACEX_PATH}/tools $@
            # exit neatly
            break
            ;;
        -o|--docker)
            # run the docker container helper script
            echo "[INFO] Running docker utility script from: ${TACEX_PATH}/docker/container.py"
            shift # past argument
            # call the python script
            python3 "${TACEX_PATH}/docker/container.py" "${@:1}"
            # exit neatly
            break
            ;;
        -v|--vscode)
            # update the vscode settings
            update_vscode_settings
            shift # past argument
            # exit neatly
            break
            ;;
        -d|--docs)
            # build the documentation
            echo "[INFO] Building documentation..."
            # retrieve the python executable
            python_exe=$(extract_python_exe)
            # install pip packages
            cd ${TACEX_PATH}/docs
            _pip_install -r requirements.txt > /dev/null
            # build the documentation
            ${python_exe} -m sphinx -b html -d _build/doctrees . _build/current
            # open the documentation
            echo -e "[INFO] To open documentation on default browser, run:"
            echo -e "\n\t\txdg-open $(pwd)/_build/current/index.html\n"
            # exit neatly
            cd - > /dev/null
            shift # past argument
            # exit neatly
            break
            ;;
        -h|--help)
            print_help
            exit 1
            ;;
        *) # unknown option
            echo "[Error] Invalid argument provided: $1"
            print_help
            exit 1
            ;;
    esac
done
