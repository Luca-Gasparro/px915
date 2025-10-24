import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion import average_diffusion_plot, overlay_average_diffusion_plot


temperature_array = np.linspace(600, 300, num=16)

avg_diffusion_coefficients_dry = average_diffusion_plot(
    npz_files=[
        "dry_diffusion_config1.npz",
        "dry_diffusion_config2.npz",
        "dry_diffusion_config3.npz",
    ],
    temperature_array=temperature_array,
    residue_name="NAP",
    is_dry=True,
)

avg_diffusion_coefficients_wet = average_diffusion_plot(
    npz_files=[
        "wet_diffusion_config1.npz",
        "wet_diffusion_config2.npz",
        "wet_diffusion_config3.npz",
    ],
    temperature_array=temperature_array,
    residue_name="NAP",
    is_dry=False,
)

overlay_average_diffusion_plot(
    dry_npz_files=[
        "dry_diffusion_config1.npz",
        "dry_diffusion_config2.npz",
        "dry_diffusion_config3.npz",
    ],
    wet_npz_files=[
        "wet_diffusion_config1.npz",
        "wet_diffusion_config2.npz",
        "wet_diffusion_config3.npz",
    ],
    temperature_array=temperature_array,
    residue_name="NAP",
)
