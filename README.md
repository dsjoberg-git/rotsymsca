# rotsymsca
Code for simulation of electromagnetic scattering from rotationally symmetric objects. It is based on [FEniCSx](https://fenicsproject.org/), an open source finite element computing platform.

## Installation

The installation instructions below are based on the instructions for installing [fe2ms](https://github.com/nwingren/fe2ms), a FE-BI hybrid code developed by Niklas Wingren. 

The code is based on FEniCSx which is available on macOS and Linux. However, installation of this package has only been tested on Ubuntu and the installation instructions are written for this. For Windows users, Linux can be run easily using Windows Subsystem for Linux (WSL). Installation instructions and more information can be found [here](https://learn.microsoft.com/en-us/windows/wsl/install). Make sure you install version WSL2.

Installation using mamba (similar to conda) is recommended. The instructions are as follows.

### Install mamba

Please follow [these](https://github.com/conda-forge/miniforge#mambaforge) instructions to install mamba. Following this, it is recommended that you create a new environment as follows ("fe2ms" can be changed to your preferred environment name).

```bash
mamba create --name rotsymsca
mamba activate rotsymsca
```

### Install FEniCSx

Having activated the fe2ms environment, the following will install fenicsx there.

```bash
mamba install fenics-dolfinx mpich petsc=*=complex*
```

### Install other Python packages

This will install other Python packages into the fe2ms environment. ```imageio``` seems to need to be installed through pip instead of mamba. The package ```miepython``` is only used to run the verification case of scattering against a sphere.

```bash
mamba install scipy matplotlib python-gmsh pyvista pyvistaqt miepython
pip install imageio[ffmpeg]
```
You also need the package ```sphericalvectorwaves```.


### Install some optional packages

Paraview is a visualization program that can be convenient, but not necessary, to use. Something seems broken in the mamba installation, but it can be installed directly in the system, rather than in the environment. Also vlc (for viewing videos) seems to be easier to install in the system.

```bash
sudo apt update
sudo apt install paraview
sudo apt install vlc
```


## Files

- ```mesh_rotsymradome.py``` sets up the mesh for two cases: an ogive radome or a sphere (PEC or material). The absorbing boundary is a PML, either spherical or cylindrical. The ogive radome can penetrate the cylindrical PML, enabling simulating a semi-infinite fuselage.
- ```rotsymsca.py``` is the main simulation code.
- ```verification.py``` runs a sphere verification case using ```miepython``` as reference code. 
- ```radome_simulations.py``` is an example of how a parameterized simulation can be set up. 

## Author

Daniel Sjöberg, Lund University. [daniel.sjoberg@eit.lth.se](mailto:daniel.sjoberg@eit.lth.se)
