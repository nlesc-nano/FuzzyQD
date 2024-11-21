#!/usr/bin/env python3
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
import numpy as np
import argparse
import os


def parse_arguments():
    """Parse command-line arguments for plotting."""
    parser = argparse.ArgumentParser(description="Plot Kappa-Energy-Phi Folded Binned Data")
    parser.add_argument("--hdf5", type=str, required=True, help="Path to the HDF5 file containing the data.")
    parser.add_argument("--output", type=str, help="Path to save the plot image (optional).")
    parser.add_argument("--energy_window", type=float, nargs=2, metavar=('min_energy', 'max_energy'),
                        help="Energy window for the plot (e.g., --energy_window -1.0 1.0).")
    return parser.parse_args()


def create_fade_black_colormap():
    """
    Create a custom colormap based on the Inferno colormap with black fading below vmin
    and white fading above vmax.
    """
    base_cmap = colormaps.get_cmap("inferno")  # Use Inferno as the base colormap

    # Add black for values below vmin and white for values above vmax
    colors = [(0, 0, 0)] + list(base_cmap(np.linspace(0, 1, 256))) + [(1, 1, 1)]  # Extend with black and white
    positions = [0.0] + list(np.linspace(0.1, 0.9, 256)) + [1.0]  # Adjust positions

    # Create the colormap
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("InfernoWithBlackWhite", list(zip(positions, colors)))
    return custom_cmap


def plot_kappa_energy_phi(hdf5_file, output_file=None, energy_window=None):
    """
    Generate a 2D color plot with kappa_ext as x-axis, energy as y-axis, and phi_folded_binned as z-values.
    """
    if not os.path.exists(hdf5_file):
        raise FileNotFoundError(f"The specified HDF5 file does not exist: {hdf5_file}")

    # Load data from HDF5 file
    with h5py.File(hdf5_file, 'r') as hdf5:
        kappa_ext = hdf5['kappa_ext'][1:]
        energy = hdf5['energy'][1:]
        phi_folded_binned = hdf5['phi_folded_binned'][:]
        tick_labels = [label.decode('utf-8') for label in hdf5['tick_labels'][:]]
        ticks = hdf5['ticks'][:]

    # Print the tick labels and tick positions
    print("Tick Labels:")
    print(tick_labels)
    print("Tick Positions:")
    print(ticks)

    # Apply energy window filtering
    if energy_window:
        min_energy, max_energy = energy_window
        energy_mask = (energy >= min_energy) & (energy <= max_energy)
        energy = energy[energy_mask]
        phi_folded_binned = phi_folded_binned[:, energy_mask]
        print(f"Using specified energy window: {min_energy} to {max_energy} eV")
    else:
        print("Using full energy range")

    # Debugging: Print array shapes
    print(f"Shape of kappa_ext: {kappa_ext.shape}")
    print(f"Shape of energy: {energy.shape}")
    print(f"Shape of phi_folded_binned: {phi_folded_binned.shape}")

    # Create meshgrid for contour plotting
    X, Y = np.meshgrid(kappa_ext, energy)
    print(f"Shape of meshgrid X: {X.shape}, Y: {Y.shape}")

    # Dynamically determine vmax and vmin
    vmax = np.max(phi_folded_binned)
    vmin = vmax / 1000  # Three orders of magnitude below vmax
    print(f"Dynamic vmin: {vmin}, vmax: {vmax}")

    # Create the custom colormap
    cmap = create_fade_black_colormap()

    # Create the plot with imshow
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("black")

    # Calculate extent for correct axis scaling
    extent = [kappa_ext.min(), kappa_ext.max(), energy.min(), energy.max()]

    # Use imshow
    img = ax.imshow(
        phi_folded_binned.T,
        aspect='auto',
        cmap=cmap,
        norm=norm,
        origin='lower',  # Align the lower-left corner with (0, 0)
        extent=extent
    )

    # Add a colorbar
    cbar = plt.colorbar(img, ax=ax, extend="both")
    cbar.set_label("Phi Folded Binned (Log Intensity)", color="black")
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.get_yticklabels(), color="black")

    # Configure axes
    ax.set_xlabel("K-path", color="black")
    ax.set_ylabel("Energy (eV)", color="black")
    ax.set_title("Kappa-Energy-Phi Folded Binned Map (Using Imshow)", color="black")
    ax.tick_params(colors="black")

    # Add tick labels and positions to the X-axis in bold
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$\\mathbf{{{label}}}$" for label in tick_labels], color="black")

    # Save or display the plot
    if output_file:
        plt.savefig(output_file, dpi=300, facecolor="white")  # Save with white background
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    args = parse_arguments()
    energy_window = tuple(args.energy_window) if args.energy_window else None
    plot_kappa_energy_phi(args.hdf5, args.output, energy_window)


if __name__ == "__main__":
    main()

