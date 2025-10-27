# Calculating the Diffusion Coefficient of Naproxen in a NAP/PVP Amorphous Solid Dispersion

Amorphous solid dispersions (ASD) are a promising formulation approach, in which drug molecules are uniformly dispersed within an amorphous polymeric matrix. Together, the drug and polymer act as an amorphous state of matter, which in turn increases the solubility of the drug, increasing efficacy and bioavaiability. Despite this, they suffer from being thermodynamically unstable, with the drug often forming amorphous or crystalline domains on timescales pertinent pharmaceutical storage time. It is known that the molecular mobility is correlated to the instability of the ASD. To this end, an important factor in considering the stability of an ASD is the diffusion coefficient of the drug molecules. In this study, we computed diffusion coefficient of naproxen in both a wet and dry structure. This was implemented through the application of Einstein's diffusion relation, and the full details of this are presented in the report.

## Installation/Setup Instructions

1. Fetch the repository using `git clone git@github.com:Luca-Gasparro/px915.git` if you want to work remotely or `git clone https://github.com/Luca-Gasparro/px915.git`. We reccomend working remotely, purely because storing the trajectories can take up a lot of storage.

2. The python package `uv` will need to be installed. This is especially important if working remotely, as `uv` acts a project manager, setting up a venv and installing the necessary packages for the project, avoiding the need to complicately work around the strict SCRTP install options. Install using `curl -LsSf https://astral.sh/uv/install.sh | sh`.

3. Go the the root directory, which contains the `pyproject.toml` file. This contains all the information regarding which python packages have been used for this project. Use `uv sync` to install these packages.

4. Activate the virtual environment in both the terminal and the python interpreter. For the terminal, run `source /storage/department/SCRTP username/px915/.venv/bin/activate`. For the python interpreter, `ctrl+shift+p` and select `Python: Select Interpreter`. If the venv does not appear in the drop down list, input the interpreter path manually, entering `/storage/department/SCRTP username/px915/.venv/bin/python`.

5. The trajectory files are too big to store on github, so contact me and we will figure out a way that works for both of us in how I can send them to you. Then ensure that the trajectories corresponding a given configuration are put into their corresponding diffusion directory. This should be simple to do as the directories are clearly labelled, however contact me if you run into trouble with this as it important that this is done properly.

## Running the scripts
There are two approaches to running the script. The first approach is too simply run the script in the `average_diffusion`. Because the diffusion coefficients are saved in an `.npz` file in each directory, the functions will skip the calculation of the mean square displacements which is path dependent. Essentially, the script will produce a plot of the average diffusion coefficient of both the wet and dry configurations. It will also plot the diffusion coefficients of the individual wet and dry configurations, so error bars can be produced. Detials of the error bar metric are described in the report if interested.

The second approach actually involves running the mean square displacement calculations for each configuration. If you really want to do this (which I encourage), first go into each directory related to diffusion and remove the `.npz` files. Then go into each `diffusion_config_i` directory, and edit the `directory` argument of the `diffusion_calculator` with your path. Then run the script. This will produce the needed `.npz` files. Move all the `.npz` files into `average_diffusion` and then run that script.

### Disclaimer
Please provide feedback on what could be better so I can incorporate it into my current work. Things I already realised:
1. Haven't checked for convergence of the diffusion coefficients - this took too much time to get done for the summer project but it's done now.
2. Fitting windows are wrong - they shouldn't go beyond half of the trajectory as moving time origins cause worse statistics.
