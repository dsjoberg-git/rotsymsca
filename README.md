# rotsymsca
Code for simulation of electromagnetic scattering from rotationally symmetric objects. It is based on [FEniCSx](https://fenicsproject.org/), an open source finite element computing platform.

## Installation

The installation instructions below are based on the instructions for installing [fe2ms](https://github.com/nwingren/fe2ms), a FE-BI hybrid code developed by Niklas Wingren. 

The code is primarily based on FEniCSx which is available on macOS and Linux. However, installation of this package has only been tested on Ubuntu and the installation instructions are written for this. For Windows users, Linux can be run easily using Windows Subsystem for Linux (WSL). Installation instructions and more information can be found [here](https://learn.microsoft.com/en-us/windows/wsl/install).

Installation using mamba (similar to conda) is recommended. The instructions are as follows.

### Install mamba

Please follow [these](https://github.com/conda-forge/miniforge#mambaforge) instructions to install mamba. Following this, it is recommended that you create a new environment as follows ("fe2ms" can be changed to your preferred environment name).

```bash
mamba create --clone base --name fe2ms
mamba activate fe2ms
```

### Install FEniCSx

Having activated the fe2ms environment, the following will install fenicsx there.

```bash
mamba install fenics-dolfinx=0.6.0 mpich petsc=*=complex*
```

### Install other Python packages

This will install other Python packages into the fe2ms environment. ```imageio``` seems to need to be installed through pip instead of mamba. 

```bash
mamba install scipy matplotlib numba python-gmsh mpi4py
pip install imageio[ffmpeg]
```

### Install some optional packages

Paraview is a visualization program that can be convenient, but not necessary, to use. Something seems broken in the mamba installation, but it can be installed directly in the system, rather than in the environment. Also vlc (for viewing videos) seems to be easier to install in the system.

```bash
sudo apt update
sudo apt install paraview
sudo apt install vlc
```

On WSL systems, OpenGL is not installed by default. This is needed by gmsh and needs to be installed. For a fresh WSL installation using Ubuntu, the following will enable OpenGL for gmsh.
```bash
sudo apt update
sudo apt upgrade
sudo apt install mesa-utils
```

## Verification

The script ```verification.py``` 
