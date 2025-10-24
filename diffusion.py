# Module to analyse the diffusion of the API within the amorphous polymer dispersion
from typeguard import typechecked
from typing import List, Tuple
import os
import re
import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.msd as msd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


@typechecked
def msd_traj_organiser(directory: str, is_dry: bool) -> List[str]:
    """Organises trajectory files based on increasing time in a file.
    This is because we need a notion of increasing simulation time
    when we calcuulate the diffusion coefficients as different times
    correspond to different temperatures within our cooling ramp
    trajectories."""

    # Collecting the trajectories on which the MSDs need to be analysed
    files = []
    selector = "dry" if is_dry else "wet"

    for filename in os.listdir(directory):
        if filename.endswith(".xtc") and selector in filename:
            full_path = os.path.join(directory, filename)
            files.append(full_path)
    # Sorts the files based on increasing numerics
    files.sort(
        key=lambda filepath: int(
            re.search(r"segment_(\d+)\.xtc", os.path.basename(filepath)).group(1)
        )
        if re.search(r"segment_(\d+)\.xtc", os.path.basename(filepath))
        else float("inf")
    )

    return files


@typechecked
def msd_temp_calculator(
    topology_file: str,
    trajectory_array: List[str],
    residue_name: str,
    cache_file_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the MSD at varous temperatures, storing them in an `.npz` file.
    Stored in this file so we don't need to run the calculation multiple times for
    one configuration. This function takes in multiple trajectories which correspond
    to different times in the cooling ramp simulation."""

    # Calculation is lengthy, so first check if it is already done. If already done
    # skip it
    if os.path.exists(cache_file_name):
        print(f"Loading cached MSD data from {cache_file_name}")
        data = np.load(cache_file_name, allow_pickle=True)
        return data["msd_array"], data["lagtimes_array"]

    msd_array = []
    lagtimes_array = []

    for traj in trajectory_array:
        u = mda.Universe(topology_file, traj)
        MSD = msd.EinsteinMSD(
            u, select=f"resname {residue_name}", msd_type="xyz", fft=True
        )
        MSD.run()

        # Coversion as MDAnalysis likes to work in angstroms - we want to work in squared centimeters
        msd_cm2 = MSD.results.timeseries / 1e16

        # Calculate lagtimes for plotting the MSDs - these will be in picoseconds
        number_of_frames = MSD.n_frames
        timestep = float(u.trajectory.dt)
        lagtime = np.arange(number_of_frames) * timestep

        msd_array.append(msd_cm2)
        lagtimes_array.append(lagtime)

    # Save results into the `.npz` file to avoid recalculating them each
    # time for each configuration.
    np.savez(cache_file_name, msd_array=msd_array, lagtimes_array=lagtimes_array)
    print(f"Saved MSD data to {cache_file_name}")

    return np.array(msd_array), np.array(lagtimes_array)


@typechecked
def msd_temp_plotter(
    msd_array: np.ndarray,
    lagtime_array: np.ndarray,
    residue_name: str,
    temperature_array: np.ndarray,
    is_dry: bool,
) -> None:
    """Plots MSD curves against time at various temperatues which the user must provide based on the cooling ramp
    .mdp file information.
    This is provided for the peer review project. Provides a linear plot and a log-log plot."""

    wet_label = "Dry" if is_dry else "Wet"

    # ------- LOG-LOG PLOT -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for lagtimes, msd_vals, temp in zip(lagtime_array, msd_array, temperature_array):
        # Plot on log-log scale, but avoid zero or negative values (which log can't handle)
        mask = (lagtimes > 0) & (msd_vals > 0)
        plt.loglog(lagtimes[mask], msd_vals[mask], label=f"{temp} K")

    plt.xlabel("Time (ps, log scale)")
    plt.ylabel(r"MSD (cm$^2$, log scale)")
    plt.title(
        f"Log-Log MSD vs Time for {wet_label} {residue_name} at Various Temperatures"
    )
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"msd_curves_various_temps_loglog_{wet_label.lower()}.png")

    # ------- LINEAR PLOT -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for lagtimes, msd_vals, temp in zip(lagtime_array, msd_array, temperature_array):
        plt.plot(lagtimes, msd_vals, label=f"{temp} K")

    plt.xlabel("Time (ps)")
    plt.ylabel(r"MSD (cm$^2$)")
    plt.title(f"MSD vs Time for {wet_label} {residue_name} at Various Temperatures")
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"msd_curves_various_temps_{wet_label.lower()}.png")

    return


@typechecked
def diffusion_coefficients(
    msd_array: np.ndarray,
    lagtime_array: np.ndarray,
    start_ps: int,
    end_ps: int,
    diffusion_file_name: str,
) -> np.ndarray:
    """Calculate diffusion coefficients based on the MSDs. Start and end
    time for fitting must be inferred manually by looking for the linear
    regime of the MSD plot. Ideally the user must not use an upper end time
    greater than half of the simulation time. I forgot about this when
    doing the summer project but for reproducibility I am not fixing this.
    The start and end time are given by the user in picoseconds to saves
    them manually converting them themselves after coming up with a valid fitting window."""

    diffusion_coeffs = []

    for msd_vals, lagtimes in zip(msd_array, lagtime_array):
        # Convert units for GROMACS-style - angstroms taken care of so just need
        # to convert picoseconds to seconds
        lagtimes_seconds = lagtimes / 1e12

        # Select fitting window
        start_index = np.searchsorted(lagtimes_seconds, start_ps / 1e12)
        end_index = np.searchsorted(lagtimes_seconds, end_ps / 1e12)

        if end_index > len(lagtimes_seconds):
            end_index = len(lagtimes_seconds)

        slope, _, _, _, _ = linregress(
            lagtimes_seconds[start_index:end_index], msd_vals[start_index:end_index]
        )

        # Einstein relation to caculate diffusion coefficients
        diff_coeff = slope / 6
        diffusion_coeffs.append(diff_coeff)

    # Save diffusion coefficients to files for the average plot
    if diffusion_file_name is not None:
        np.savez(diffusion_file_name, diffusion_array=diffusion_coeffs)
        print(f"Saved diffusion coefficients to {diffusion_file_name}")

    return np.array(diffusion_coeffs)


@typechecked
def plot_diffusion_vs_temperature(
    temperature_array: np.ndarray,
    diffusion_array: np.ndarray,
    residue_name: str,
    is_dry: bool,
) -> None:
    """Plot of the diffusion coefficient against temperature for
    a given wet or dry configuration."""

    wet_label = "Dry" if is_dry else "Wet"
    plt.figure(figsize=(8, 6))

    plt.plot(temperature_array, diffusion_array, marker="o", linestyle="None")

    # Reverse x-axis: higher temp on left
    plt.gca().invert_xaxis()

    plt.xlabel("Temperature (K)")
    plt.ylabel(r"Diffusion Coefficient (cm$^2$)")
    plt.title(f"Diffusion Coefficients vs Temperature for {wet_label} {residue_name}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"diffusion_vs_temp_{wet_label.lower()}.png")

    return


@typechecked
def average_diffusion_plot(
    npz_files: List[str],
    temperature_array: np.ndarray,
    residue_name: str,
    is_dry: bool,
) -> np.ndarray:
    all_diffusions = []

    """Plots the average diffusion coefficients of the API at each temperature. Also plots the diffusion
    coefficient of the API at each temperature for the given configurations. Errors are taken to be
    maximum difference between the average diffusion coefficient and the diffusion coefficient for the 
    given configurations."""

    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        diffusion_array = data["diffusion_array"]
        all_diffusions.append(diffusion_array)

    all_diffusions_array = np.array(all_diffusions)
    avg_diffusions = np.mean(all_diffusions_array, axis=0)

    # Compute symmetric error bars as the maximum distance from average
    max_distance = np.max(np.abs(all_diffusions_array - avg_diffusions), axis=0)

    # Plotting
    label = "Dry" if is_dry else "Wet"
    plt.figure(figsize=(8, 6))
    # Plot individual points
    for i, temp in enumerate(temperature_array):
        plt.scatter(
            [temp] * all_diffusions_array.shape[0],
            all_diffusions_array[:, i],
            color="red",
            s=10,
            zorder=2,
            label="Individual Configurations" if i == 0 else None,
        )

    # Plot average line
    # Plot average diffusion with symmetric error bars
    plt.errorbar(
        temperature_array,
        avg_diffusions,
        yerr=max_distance,
        fmt="ko",
        capsize=3,
        markersize=5,
        label="Average Diffusion Coefficient",
        zorder=3,
    )

    plt.gca().invert_xaxis()
    plt.xlabel("Temperature (K)")
    plt.ylabel(r"Average Diffusion Coefficient (cm$^2$/s)")
    plt.title(f"Average Diffusion Coefficient vs Temperature ({label} {residue_name})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"average_diffusion_vs_temp_{label.lower()}.png")
    plt.close()

    return avg_diffusions


@typechecked
def msd_calculator(
    directory: str,
    topology_file: str,
    residue_name: str,
    cache_file_name: str,
    temperature_array: np.ndarray,
    is_dry: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Chains MSD related functions to obtain an MSD plot for a given wet or dry
    configuration. The user can then inspect them manually to obtain the fitting
    window needed for the diffusion coefficient calculation."""

    trajs = msd_traj_organiser(directory=directory, is_dry=is_dry)
    msd_array, lagtime_array = msd_temp_calculator(
        topology_file=topology_file,
        trajectory_array=trajs,
        residue_name=residue_name,
        cache_file_name=cache_file_name,
    )
    msd_temp_plotter(
        msd_array=msd_array,
        lagtime_array=lagtime_array,
        residue_name=residue_name,
        temperature_array=temperature_array,
        is_dry=is_dry,
    )

    return msd_array, lagtime_array


@typechecked
def diffusion_calculator(
    msd_array: np.ndarray,
    lagtime_array: np.ndarray,
    start_ps: int,
    end_ps: int,
    temperature_array: np.ndarray,
    residue_name: str,
    is_dry: bool,
    diffusion_file_name: str,
) -> np.ndarray:
    """Chains diffusion related functions to calculate the diffusion coefficients
    based on the provided fitting windows."""
    diff_coeffs = diffusion_coefficients(
        msd_array=msd_array,
        lagtime_array=lagtime_array,
        start_ps=start_ps,
        end_ps=end_ps,
        diffusion_file_name=diffusion_file_name,
    )
    plot_diffusion_vs_temperature(
        temperature_array=temperature_array,
        diffusion_array=diff_coeffs,
        residue_name=residue_name,
        is_dry=is_dry,
    )

    return diff_coeffs


@typechecked
def overlay_average_diffusion_plot(
    dry_npz_files: List[str],
    wet_npz_files: List[str],
    temperature_array: np.ndarray,
    residue_name: str,
) -> None:
    """Plots the average diffusion coefficients of the API at each temperature. Also plots the diffusion
    coefficient of the API at each temperature for the given configurations. Errors are taken to be
    maximum difference between the average diffusion coefficient and the diffusion coefficient for the
    given configurations. Both the wet and dry diffusion coefficents are on the same plot
    for presentational purposes. An inset plot has been produced for the temperatures of biological interest."""

    # Dry data
    dry_diffs = []
    for f in dry_npz_files:
        data = np.load(f, allow_pickle=True)
        dry_diffs.append(data["diffusion_array"])
    dry_arr = np.array(dry_diffs)
    dry_avg = np.mean(dry_arr, axis=0)
    dry_err = np.max(np.abs(dry_arr - dry_avg), axis=0)

    # Wet data
    wet_diffs = []
    for f in wet_npz_files:
        data = np.load(f, allow_pickle=True)
        wet_diffs.append(data["diffusion_array"])
    wet_arr = np.array(wet_diffs)
    wet_avg = np.mean(wet_arr, axis=0)
    wet_err = np.max(np.abs(wet_arr - wet_avg), axis=0)

    # Reference data from Zhong et al.
    ref_wet_D = 4.0e-10
    ref_temp = 313

    plt.figure(figsize=(8, 6))

    # Pot dry individual points
    for i, temp in enumerate(temperature_array):
        plt.scatter(
            [temp] * dry_arr.shape[0],
            dry_arr[:, i],
            color="red",
            s=10,
            zorder=2,
            label="Dry Configurations" if i == 0 else None,
        )

    # Plot wet individual points
    for i, temp in enumerate(temperature_array):
        plt.scatter(
            [temp] * wet_arr.shape[0],
            wet_arr[:, i],
            color="blue",
            s=10,
            zorder=2,
            label="Wet Configurations" if i == 0 else None,
        )

    # Plot reference point
    plt.plot(
        ref_temp,
        ref_wet_D,
        linestyle="None",
        label="Reference Value (Wet)",
        color="green",
        marker="x",
    )

    # Plot dry diffusion coefficient average
    plt.errorbar(
        temperature_array,
        dry_avg,
        yerr=dry_err,
        fmt="s",  # square
        color="red",
        ecolor="black",
        markersize=5,
        capsize=3,
        label="Dry Average",
        zorder=3,
    )

    # Plot wet diffusion coefficient average
    plt.errorbar(
        temperature_array,
        wet_avg,
        yerr=wet_err,
        fmt="s",  # square
        color="blue",
        ecolor="black",
        markersize=5,
        capsize=3,
        label="Wet Average",
        zorder=3,
    )
    plt.gca().invert_xaxis()
    plt.xlabel("Temperature (K)", fontsize=15)
    plt.ylabel(r"Diffusion Coefficient (cm$^2$/s)", fontsize=15)
    plt.title(f"Diffusion Coefficient vs Temperature ({residue_name})", fontsize=15)
    plt.tick_params(axis="both", labelsize=15)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(fontsize=12)

    # Inset for last 3 temperatures
    last_temps = [340, 320, 300]
    indices = [np.where(temperature_array == t)[0][0] for t in last_temps]

    ax_inset = inset_axes(
        plt.gca(),
        width="40%",
        height="40%",
        loc="upper right",
        bbox_to_anchor=(0, -0.35, 1, 1),
        bbox_transform=plt.gca().transAxes,
    )

    # Plot dry points in inset
    for i in indices:
        ax_inset.scatter(
            [temperature_array[i]] * dry_arr.shape[0],
            dry_arr[:, i],
            color="red",
            s=10,
            zorder=2,
        )

    # Plot wet points in inset
    for i in indices:
        ax_inset.scatter(
            [temperature_array[i]] * wet_arr.shape[0],
            wet_arr[:, i],
            color="blue",
            s=10,
            zorder=2,
        )

    # Plot reference value in inset
    ax_inset.plot(
        ref_temp,
        ref_wet_D,
        marker="x",
        linestyle="None",
        color="green",
    )

    # Plot dry average in inset
    ax_inset.errorbar(
        temperature_array[indices],
        dry_avg[indices],
        yerr=dry_err[indices],
        fmt="s",
        color="red",
        ecolor="black",
        markersize=5,
        capsize=3,
        zorder=3,
    )

    # Plot wet average in inset
    ax_inset.errorbar(
        temperature_array[indices],
        wet_avg[indices],
        yerr=wet_err[indices],
        fmt="s",
        color="blue",
        ecolor="black",
        markersize=5,
        capsize=3,
        zorder=3,
    )

    ax_inset.invert_xaxis()
    ax_inset.set_xlabel("Temperature (K)", fontsize=10)
    ax_inset.set_ylabel(r"Diffusion Coefficient (cm$^2$/s)", fontsize=10)
    ax_inset.tick_params(axis="both", which="major", labelsize=10)
    ax_inset.grid(True, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("overlay_diffusion_vs_temp.png")
    return
