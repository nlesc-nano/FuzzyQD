#!/usr/bin/env python3
import sys
import h5py
import os
import re 
import glob
import time
import argparse
import pickle
import numpy as np

# Conversion constants
HARTREE = 27.2107

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Merge PDOS and optionally BSE pickle files.")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing the files.")
    parser.add_argument("--project", type=str, required=True, help="Project name as specified in input files.")
    parser.add_argument("-bse", action="store_true", help="Process BSE pickle files.")
    return parser.parse_args()

def extract_homo_energy(pdos_file):
    """
    Extract the HOMO (E(Fermi)) energy from the first line of a .pdos file.

    Parameters:
    -----------
    pdos_file : str
        Path to the .pdos file.

    Returns:
    --------
    float
        The extracted HOMO energy in atomic units (a.u.).
    """
    with open(pdos_file, 'r') as f:
        first_line = f.readline()
        match = re.search(r"E\(Fermi\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", first_line)
        if match:
            return float(match.group(1))
        else:
            raise ValueError("Unable to extract E(Fermi) from .pdos file.")

def E_bin(folded_states, E_list, State_0, dE):
    """
    Bins folded states by energy levels, summing states within each energy bin.

    Parameters:
    - folded_states: 2D numpy array, the folded states matrix with shape (N_states, N_kappa).
    - E_list: 2D numpy array, array where the first column represents state numbers and the second column represents energies.
    - State_0: int, the reference state number for binning.
    - dE: float, the energy bin width.

    Returns:
    - bin_edges: 1D numpy array, the edges of each energy bin.
    - E_binned_folded_states: 2D numpy array, the summed folded states for each energy bin.
    """

    start_time = time.time()
    print("Starting energy binning...")

    # Determine indices for the folded states relative to the energy list
    E_list_state_0 = int(E_list[0, 0])
    N_states = folded_states.shape[0]
    clip_min, clip_max = State_0 - E_list_state_0, State_0 - E_list_state_0 + N_states
    E_list_states = E_list[int(clip_min):int(clip_max), :]
    E = E_list_states[:, 1]

    # Determine bin edges based on energy range and bin width
    bin_edges = np.arange(E.min(), E.max() + dE, dE)
    print(f"Bin edges calculated with range {E.min()} to {E.max()} and width {dE}")

    # Assign each energy value to a bin number
    bin_numbers = np.digitize(E, bin_edges)

    # Initialize array to store summed folded states for each bin
    E_binned_folded_states = np.zeros((len(bin_edges) - 1, folded_states.shape[1]))

    # Timing individual bin processing
    binning_start = time.time()
    for i in range(len(bin_edges) - 1):
        bin_mask = bin_numbers == (i + 1)
        E_binned_folded_states[i] = np.sum(folded_states[bin_mask], axis=0)
    print(f"Energy binning completed in {time.time() - binning_start:.4f} seconds")

    print(f"Total E_bin function completed in {time.time() - start_time:.4f} seconds")
    return bin_edges, E_binned_folded_states

def hdf5_pdos_out(project, state_min, pdos, pdos_binned, E_array, Elements):
    """
    Save PDOS (Projected Density of States) data into an HDF5 file with cumulative 
    and fractional cumulative sums for each element.

    Parameters:
    -----------
    project : str
        Project name used for naming the output HDF5 file.
    state_min : int
        Minimum state index corresponding to the PDOS data.
    pdos : np.ndarray
        Original PDOS data with shape (N_states, N_elements).
    pdos_binned : np.ndarray
        Binned PDOS data with shape (N_energy, N_elements).
    E_array : np.ndarray
        Energy array corresponding to the binned PDOS data.
    Elements : list
        List of element identifiers corresponding to the PDOS columns.

    Outputs:
    --------
    HDF5 File:
        An HDF5 file is created with datasets for original PDOS, binned PDOS, 
        cumulative PDOS, and fractional cumulative PDOS for each element.

    Notes:
    ------
    - Rows with zero PDOS are handled to avoid division errors in fractional calculations.
    - The output filename is generated based on the project name and state range.

    Example File Structure:
    ------------------------
    - Elements: List of element identifiers.
    - States: Array of state numbers.
    - Energy: Array of energy values.
    - pdos_X: Original PDOS for element X.
    - pdos_binned_X: Binned PDOS for element X.
    - pdos_binned_acc_X: Cumulative binned PDOS for element X.
    - frac_pdos_binned_acc_X: Fractional cumulative binned PDOS for element X.
    """
    start_time = time.time()

    # Initialization and file naming
    N_states = np.size(pdos, axis=0)
    N_energy = np.size(pdos_binned, axis=0)
    N_elements = np.size(pdos, axis=1)
    state_nr = np.linspace(state_min, state_min + N_states - 1, N_states)
    N_i = state_min
    N_f = N_i + N_states - 1
    fname = f"{project}_pdos_States_{int(N_i)}_{int(N_f)}.h5"

    # Compute cumulative PDOS
    pdos_acc = np.cumsum(pdos_binned, axis=1)

    # Compute fractional cumulative PDOS
    pdos_sum = np.sum(pdos_binned, axis=1)
    rows_with_zero_pdos = np.where(pdos_sum == 0)[0]
    fractional_pdos = pdos_acc.copy()
    for i in range(N_energy):
        if i not in rows_with_zero_pdos:
            fractional_pdos[i, :] /= pdos_acc[i, N_elements - 1]

    # Write data to HDF5 file
    with h5py.File(fname, 'w') as hdf5:
        hdf5.create_dataset('Elements', data=Elements)
        hdf5.create_dataset('States', data=state_nr)
        hdf5.create_dataset('Energy', data=E_array)

        for element in range(N_elements):
            hdf5.create_dataset(f'pdos_{element}', data=pdos[:, element])
            hdf5.create_dataset(f'pdos_binned_{element}', data=pdos_binned[:, element])
            hdf5.create_dataset(f'pdos_binned_acc_{element}', data=pdos_acc[:, element])
            hdf5.create_dataset(f'frac_pdos_binned_acc_{element}', data=fractional_pdos[:, element])

    end_time = time.time()
    print(f"HDF5 file '{fname}' created successfully in {end_time - start_time:.2f} seconds.")

def hdf5_output(project, k_path_names, kappa_ticks, kappa, state_nr, bse_folded_states, bse_folded_states_binned, E_array):
    """
    Writes BSE data, kappa path, state identifiers, and energy data to an HDF5 file.

    Parameters:
    - project: str, base name of the project used to name the output file.
    - k_path_names: list, names of high-symmetry points along the k-path.
    - kappa_ticks: array-like, positions of k-path ticks corresponding to high-symmetry points.
    - kappa: array-like, array of wavenumbers along the k-path.
    - state_nr: array-like, array of state numbers.
    - bse_folded_states: 2D numpy array, folded BSE states for each state in state_nr.
    - bse_folded_states_binned: 2D numpy array, binned BSE folded states.
    - E_array: array-like, array of energy values.

    Returns:
    - None
    """

    start_time = time.time()
    N_states = np.size(state_nr)
    N_kappa = np.size(kappa)
    d_kappa = kappa[1] - kappa[0]
    N_i = state_nr[0]
    N_f = state_nr[-1]
    fname = f"{project}_bse_States_{int(N_i)}_{int(N_f)}.h5"

    # Extend kappa for visualization or processing purposes
    kappa_ext = np.concatenate((kappa - d_kappa / 2, [kappa[-1] + d_kappa / 2]))

    print(f"Writing BSE data to HDF5 file: {fname}")
    with h5py.File(fname, 'w') as hdf5:
        hdf5.create_dataset('tick_labels', data=k_path_names)
        hdf5.create_dataset('ticks', data=kappa_ticks)
        hdf5.create_dataset('kappa', data=kappa)
        hdf5.create_dataset('kappa_ext', data=kappa_ext)
        hdf5.create_dataset('state_identifier', data=state_nr)
        hdf5.create_dataset('phi_folded', data=bse_folded_states)
        hdf5.create_dataset('phi_folded_binned', data=bse_folded_states_binned)
        hdf5.create_dataset('energy', data=E_array)

    print(f"HDF5 data written to {fname} in {time.time() - start_time:.4f} seconds")

def find_and_sort_pdos_files(folder):
    """
    Find all .pdos files in the specified folder and sort them by the numeric value in 'k*'.

    Parameters:
    -----------
    folder : str
        Path to the folder containing the .pdos files.

    Returns:
    --------
    list
        Sorted list of .pdos file paths.
    """
    pdos_files = glob.glob(os.path.join(folder, "*.pdos"))

    # Extract numeric part from 'k*' in the filenames and sort by it
    def extract_k_value(filename):
        match = re.search(r"k(\d+)", filename)
        return int(match.group(1)) if match else float('inf')  # Handle files without 'k*'

    sorted_pdos_files = sorted(pdos_files, key=extract_k_value)
    return sorted_pdos_files


def determine_pkl_params(folder, project, input_pkl):
    """
    Determine `pkl_step` and `pkl_N` dynamically based on filenames.

    Parameters:
    -----------
    folder : str
        Path to the folder containing the files.
    project : str
        Project name for identifying files.
    input_pkl : str
        Base name for pickle files.

    Returns:
    --------
    tuple: (int, int, int)
        - Minimum state index.
        - Step size for state indices.
        - Number of pickle files to process.
    """
    pkl_files = sorted(glob.glob(os.path.join(folder, f"{project}{input_pkl}_*.pkl")))
    indices = [int(f.split('_')[-2]) for f in pkl_files]  # Extract starting indices
    indices.sort()
    pkl_step = indices[1] - indices[0] if len(indices) > 1 else 0
    pkl_N = len(indices)
    state_min = indices[0] if indices else 0
    return state_min, pkl_step, pkl_N

def process_pdos(folder, project, state_min, pkl_step, pkl_N):
    """
    Read and process PDOS files.

    Parameters:
    -----------
    folder : str
        Path to the folder containing the PDOS files.
    project : str
        Project name for identifying PDOS files.
    state_min : int
        Minimum state index for energy binning.
    pkl_step : int
        Step size for the state index.
    pkl_N : int
        Number of PDOS files to process.

    Returns:
    --------
    tuple: (np.ndarray, np.ndarray)
        - Energies array from the first `.pdos` file.
        - Array of binned PDOS data.
    """
    start_time = time.time()

    # Find all .pdos files in the folder
    pdos_files = find_and_sort_pdos_files(folder)

    # Ensure files exist
    if not pdos_files:
        raise FileNotFoundError("No .pdos files found in the specified folder.")

    # Extract the HOMO (Fermi) energy from the first .pdos file
    homo_energy = extract_homo_energy(pdos_files[0])
    print(f"Extracted HOMO energy (E(Fermi)): {homo_energy * HARTREE} eV")

    # Read atom types directly from .pdos files
    elements = []
    for file in pdos_files:
        with open(file, 'r') as f:
            elements.append(f.readline().split()[6])

    # Read energy data and PDOS values
    energies = np.loadtxt(pdos_files[0], usecols=(0, 1))
    xs = [np.loadtxt(file) for file in pdos_files]
    ys = np.stack([np.sum(x[:, 3:], axis=1) for x in xs]).transpose()

    # Set print options to display all numbers in the array
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    # Shift energies so that HOMO is at 0
    energies[:, 1] -= homo_energy
    print(f"Shifted energies so that HOMO is at 0.")
    # Print the orbital energies
    print("Orbital energies (in eV)")
    print(energies[:, 1] * HARTREE)

    # Process binned PDOS
    energies_0 = int(energies[0, 0])
    clip_min = state_min - energies_0
    clip_max = clip_min + pkl_step * pkl_N
    ys_clipped = ys[int(clip_min):int(clip_max), :]

    # Refine energy bin width dynamically
    dE = (energies[:, 1].max() - energies[:, 1].min()) / len(energies)
    print(f"Refined energy bin width: {dE}")
    pdos_E_array, pdos_binned = E_bin(ys_clipped, energies, state_min, dE)
    pdos_E_array -= dE / 2
    pdos_E_array = pdos_E_array[1:]

    # Save PDOS to HDF5
    hdf5_pdos_out(project, state_min, ys_clipped, pdos_binned, pdos_E_array * HARTREE, elements)

    end_time = time.time()
    print(f"Processed {len(pdos_files)} PDOS files in {end_time - start_time:.2f} seconds.")
    return energies, pdos_binned

def process_bse(folder, project, state_min, pkl_step, pkl_N, input_pkl, energies):
    """
    Process BSE pickle files and perform energy binning.

    Parameters:
    -----------
    folder : str
        Path to the folder containing the files.
    project : str
        Project name for identifying files.
    state_min : int
        Minimum state index for energy binning.
    pkl_step : int
        Step size for the state index.
    pkl_N : int
        Number of pickle files to process.
    input_pkl : str
        Base name for BSE pickle files.
    energies : np.ndarray
        Energy array from PDOS processing.
    """
    state_nr = np.array([])
    bse_map_load = []

    for i in range(pkl_N):
        state_i = state_min + pkl_step * i
        state_f = state_i + pkl_step - 1
        file = f"{project}{input_pkl}_{state_i}_{state_f}.pkl"
        filepath = os.path.join(folder, file)

        with open(filepath, 'rb') as f:
            array1, array2 = pickle.load(f)
            state_nr = np.hstack((state_nr, array1))
            bse_map_load.append(array2)

    bse_map = np.vstack(bse_map_load)

    # Refine energy bin width dynamically
    dE = (energies[:, 1].max() - energies[:, 1].min()) / len(energies)
    print(f"Refined energy bin width: {dE}")
    bse_E_array, bse_map_binned = E_bin(bse_map, energies, state_min, dE)

    k_path_file = os.path.join(folder, f"{project}_bse_k_path.pkl")
    with open(k_path_file, 'rb') as f:
        k_path_names, kappa_ticks, kappa = pickle.load(f)

    hdf5_output(project, k_path_names, kappa_ticks, kappa, state_nr, bse_map.T, bse_map_binned.T, bse_E_array * HARTREE)

def main():
    args = parse_arguments()
    os.chdir(args.folder)

    # Input/output configuration
    in_out_data = {"input_pkl": "_bse_States"}

    # Determine parameters dynamically
    state_min, pkl_step, pkl_N = determine_pkl_params(args.folder, args.project, in_out_data["input_pkl"])

    # Process PDOS files (mandatory)
    energies, _ = process_pdos(args.folder, args.project, state_min, pkl_step, pkl_N)

    # Process BSE files (optional)
    if args.bse:
        process_bse(args.folder, args.project, state_min, pkl_step, pkl_N, in_out_data["input_pkl"], energies)

if __name__ == "__main__":
    main()

