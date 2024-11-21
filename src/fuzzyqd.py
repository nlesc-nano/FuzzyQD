#!/usr/bin/env python3
import os
import yaml
import shutil
from glob import glob
import sys
import re
import numpy as np
import funcs as bse
import matplotlib.pyplot as plt


def main():
    # Load Input Data from YAML
    if len(sys.argv) > 1:
        yaml_file_path = sys.argv[1]
    else:
        yaml_file_path = 'input_parameters.yaml'

    with open(yaml_file_path, 'r') as file:
        params = yaml.safe_load(file)

    # Check if block processing is required
    if 'blocks' in params['files']:
        handle_blocks(params, yaml_file_path)
        sys.exit(0)  # Exit after creating the folders

    # If no blocks are defined, continue with the main BSE calculations
    perform_bse_calculations(params)


def handle_blocks(params, yaml_file_path):
    """Handle block folder creation and processing based on input YAML."""
    folder_cubes = params['files']['folder_cubes']
    project_name = params['files']['Project']
    num_blocks = params['files']['blocks']

    # Navigate to cube files folder
    os.chdir(folder_cubes)

    # List all cube files and calculate block distribution
    cube_files = sorted(glob("*.cube"))
    total_cubes = len(cube_files)
    cubes_per_block = total_cubes // num_blocks
    extra_cubes = total_cubes % num_blocks

    for block_index in range(num_blocks):
        start_idx = block_index * cubes_per_block + min(block_index, extra_cubes)
        end_idx = start_idx + cubes_per_block + (1 if block_index < extra_cubes else 0)
        block_files = cube_files[start_idx:end_idx]

        # Create block folder and move relevant files
        block_folder = f"{project_name}_block_{block_index + 1}"
        os.makedirs(block_folder, exist_ok=True)

        for cube_file in block_files:
            shutil.move(cube_file, os.path.join(block_folder, cube_file))

        # Copy and modify the YAML file for the specific block
        block_yaml_path = os.path.join(block_folder, 'input_parameters.yaml')
        shutil.copy(yaml_file_path, block_yaml_path)
        modify_yaml_for_block(block_yaml_path, params, block_files, block_index, block_folder)

        # Create Slurm batch script in each block folder
        create_slurm_script(block_folder, block_yaml_path)

        print(f"Created '{block_folder}' with {len(block_files)} cube files, "
              f"modified YAML file '{block_yaml_path}', and Slurm script.")

    print("Block folders, YAML files, and Slurm scripts created. "
          "To run BSE, navigate to each block folder and submit the Slurm script with `sbatch run_bse.slurm`.")

def modify_yaml_for_block(block_yaml_path, params, block_files, block_index, block_folder):
    """Modify the YAML configuration for a specific block."""
    with open(block_yaml_path, 'r+') as yaml_file:
        block_params = yaml.safe_load(yaml_file)
        block_params['files']['folder_cubes'] = '.'
        block_params['files'].pop('blocks', None)
        block_params['files']['block_index'] = block_index + 1

        cube_files_in_block = sorted(glob(os.path.join(block_folder, "*.cube")))
        if not cube_files_in_block:
            raise ValueError(f"No .cube files found in block folder '{block_folder}'")
        first_cube_file = os.path.basename(cube_files_in_block[0])
        match = re.search(r'WFN_0(\d+)', first_cube_file)
        if match:
            first_cube_index = int(match.group(1))
            block_params['files']['cube_0'] = first_cube_index
        else:
            raise ValueError(f"Could not extract cube index from filename: {first_cube_file}")

        block_params['files']['N_cube'] = len(block_files)
        block_params.pop('k_path', None)

        yaml_file.seek(0)
        yaml.dump(block_params, yaml_file, sort_keys=False)
        yaml_file.truncate()

        yaml_file.write("\nk_path:\n")
        yaml_file.write(f"  names: {params['k_path']['names']}\n")
        yaml_file.write("  points:\n")
        for point in params['k_path']['points']:
            yaml_file.write(f"    - {point}\n")

def create_slurm_script(block_folder, block_yaml_path):
    """Create a Slurm batch script for each block folder."""
    slurm_script_path = os.path.join(block_folder, 'run_bse.slurm')
    with open(slurm_script_path, 'w') as slurm_file:
        slurm_file.write("#!/bin/bash\n")
        slurm_file.write("#SBATCH --account=euhpc_r02_106\n")
        slurm_file.write("#SBATCH --partition=dcgp_usr_prod\n")
        slurm_file.write("#SBATCH --job-name=FuzzyQD\n")
        slurm_file.write("#SBATCH --time=1-00:00:00\n")
        slurm_file.write("#SBATCH --nodes=1\n")
        slurm_file.write("#SBATCH --ntasks-per-node=1\n")
        slurm_file.write("#SBATCH --cpus-per-task=112\n")
        slurm_file.write("#SBATCH --output=%x-%j.out\n")
        slurm_file.write("#SBATCH --error=%x-%j.err\n\n")
        slurm_file.write("srun --ntasks=112 python bse_main_new.py input_parameters.yaml\n")

def perform_bse_calculations(params):
    """Perform the main BSE calculations based on the input parameters."""
    # Conversion factors and other parameters
    b2a = params.get('conversion', {}).get('bohr_to_angstrom', 0.529177)
    hartree = params.get('conversion', {}).get('hartree', 27.2107)
    a = params['lattice']['a']
    clip = params['clipping'].get('clip', False)
    if clip:
        size_clip = params['clipping']['size_clip'] / b2a
    k_unit = 2 * np.pi / (a / b2a)
    dk = params['reciprocal_space'].get('dk', 0.005)
    dE = params.get('energy_binning', {}).get('dE', 0.0125) / hartree

    if params['settings']['fcc']:
        for direction in bse.e_k_dict:
            direction['N_bun'] = 2
        k_unit *= 2

    calculation_data = {'latt_par': a, 'k_unit': k_unit, 'dx': 1, 'clip': clip, 'frame': size_clip if clip else None}

    # Define the k-path
    k_path_names = params['k_path']['names']
    k_path_points = np.array(params['k_path']['points'])
    folder_cubes = params['files']['folder_cubes']

    # Set folder and file path information
    os.chdir(folder_cubes)
    file_specifiers = {
        'Project': params['files']['Project'],
        'cube_0': params['files'].get('cube_0', 5542),
        'N_cube': params['files'].get('N_cube', 300),
        'State': 'STATES',
        'WFN': '-WFN_0',
        'Addition': '_1-1_0',
        'h5_file': 'cubedata_inp',
        'Energy': 'E',
        'h5_Energy': 'e_inp',
        'extension': ['cube', 'h5', 'txt']
    }

    # If 'block_index' exists in params['files'], include it in the log file name
    block_index = params['files'].get('block_index', None)
    if block_index is not None:
        log_file = f"{file_specifiers['Project']}_{file_specifiers['State']}_{block_index}.log"
    else:
        log_file = f"{file_specifiers['Project']}_{file_specifiers['State']}.log"

    k_path, kappa_path = [], []

    for s in range(k_path_points.shape[0] - 1):
        k_segment = bse.k_path_segment(k_path_points[s], k_path_points[s + 1], bse.e_k_dict, dk)
        k_path.append(k_segment)

    kappa_path, kappa_ticks = bse.organize_output_path(k_path, k_path_points)

    # Determine expansion in Bloch states
    cube_input = params['settings']['cube_input']
    h5_input = not cube_input

    if cube_input:
        bse_folded_states, state_nr, files_processed = bse.bse_cube(file_specifiers, log_file, k_path, kappa_path, calculation_data)
    elif h5_input:
        bse_folded_states, state_nr, files_processed = bse.bse_h5(file_specifiers, log_file, k_path, kappa_path, calculation_data)

    # Post-processing
    project = file_specifiers['Project']
    if files_processed:
        if not params['settings']['re_run']:
            bse.write_path(project, k_path_names, kappa_ticks, kappa_path)
        bse.write_bse_folded(project, state_nr, bse_folded_states)
        bse.log_output("Bloch state expansion saved as NumPy array", log_file)

if __name__ == "__main__":
    main()

