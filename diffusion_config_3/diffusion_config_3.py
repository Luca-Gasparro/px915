import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion import msd_calculator, diffusion_calculator

temperature_array = np.linspace(600, 300, num=16)

# ------ MSD CALCULATION CONFIG 3 DRY --------------------------------------------------------
msd_dry_config3, lagtimes_dry_config3 = msd_calculator(
    directory="/storage/chem/phuqdw/px915/diffusion_config_3",
    topology_file="dry_cooling_ramp.tpr",
    residue_name="NAP",
    cache_file_name="dry_msd_config3.npz",
    temperature_array=temperature_array,
    is_dry=True,
)

# ------ DIFFUSION CALCULATION CONFIG 3 DRY --------------------------------------------------
D_dry = diffusion_calculator(
    msd_array=msd_dry_config3,
    lagtime_array=lagtimes_dry_config3,
    start_ps=2000,
    end_ps=6000,
    temperature_array=temperature_array,
    residue_name="NAP",
    is_dry=True,
    diffusion_file_name="dry_diffusion_config3.npz",
)

# ------ MSD CALCULATION CONFIG 3 WET --------------------------------------------------------
msd_wet_config3, lagtimes_wet_config3 = msd_calculator(
    directory="/storage/chem/phuqdw/px915/diffusion_config_3",
    topology_file="wet_cooling_ramp.tpr",
    residue_name="NAP",
    cache_file_name="wet_msd_config3.npz",
    temperature_array=temperature_array,
    is_dry=False,
)

# ------ DIFFUSION CALCULATION CONFIG 3 WET --------------------------------------------------
D_wet = diffusion_calculator(
    msd_array=msd_wet_config3,
    lagtime_array=lagtimes_wet_config3,
    start_ps=2000,
    end_ps=6000,
    temperature_array=temperature_array,
    residue_name="NAP",
    is_dry=False,
    diffusion_file_name="wet_diffusion_config3.npz",
)

# Print to get a feel of the size of the diffusion coefficients
print(D_dry)
print(D_wet)
