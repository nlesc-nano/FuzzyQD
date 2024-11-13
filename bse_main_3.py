import numpy as np
import matplotlib.pyplot as plt
import os
import bse_def_3 as bse

#%% Input data ###########

# Conversion bohr to angstrom
b2a = 0.529177
hartree = 27.2107
#real space lattice parameter in Angstrom
a = 6.0386
# Data for clipping the cube file, including the width of the frame in Angstrom to keep
clip = True
frame = 4 / b2a
# Reciprocal space lattice unit - anticipating real space distances in bohr. For fcc, the k_unit is automatically adjusted.
k_unit = 2 * np.pi / (a / b2a) 
# Set reciprocal space resolution in units 2 np.pi / a
dk = 0.005
# Set resolution for energy binning in eV
dE = 0.0125 / hartree
# Define the k-path - measure reciprocal space coordinates in units of 2 np.pi / a for bcc and 4 np.pi / a for fcc.
k_path_names = ['L', 'G', 'Xx', 'W', 'K', 'G', 'Xy']
k_path_points = np.array([[0.250, 0.250, 0.250],\
                          [0.000, 0.000, 0.000],\
                          [0.500, 0.000, 0.000],\
                          [0.500, 0.250, 0.000],\
                          [0.375, 0.375, 0.000],\
                          [0.000, 0.000, 0.000],\
                          [0.000, 0.500, 0.000]])

#%% Set folder and file path information

""" Anticipated file format for states is a CP2K output file Project_States-WFN_02706_1-1_0, with 
    Project the project name and 2706 the number of the state. Energies are supposed to be stored
    in a similar file with name Project_States_E.txt that provides energies in Hartree.
"""

folder_cubes = '/dodrio/scratch/projects/starting_2024_023/BSE/InAs/3.0nm/States'
file_specifiers = {'Project': 'InAs1116', 'State': 'STATES', 'WFN': '-WFN_0', 'Addition': '_1-1_0',\
                   'h5_file': 'cubedata_inp', 'Energy': 'E', 'h5_Energy': 'e_inp', 'cube_0': 5542, 'N_cube': 300,\
                   'extension': ['cube', 'h5', 'txt']}
re_run = False
np_array_dump = True
h5_output = False
E_binning = False
fcc = True
plot = False
cube_input = True
h5_input = not(cube_input)
os.chdir(folder_cubes)

log_file = file_specifiers['Project'] + '_' + file_specifiers['State'] + '6' +  '.log' 

if fcc:
    for direction in bse.e_k_dict:
        direction['N_bun'] = 2
    k_unit *= 2

calculation_data = {'latt_par': a, 'k_unit': k_unit, 'dx': 1, 'clip': clip, 'frame': frame}

#%% Create dictionary with path data
""" Translates the path into k_path, an array of dictionaries that each contain the essential data of a segment 
of the path defined through reciprocal space. Information important for the entire path is included in the dictionary
k_structure.
"""
k_path = []
segments = k_path_points.shape[0] - 1
for s in range(segments):
    k_1 = k_path_points[s, :]
    k_2 = k_path_points[(s + 1), :]
    k_segment = bse.k_path_segment(k_1, k_2, bse.e_k_dict, dk)
    k_path.append(k_segment)
kappa_path, kappa_ticks = bse.organize_output_path(k_path, k_path_points)
# Set up array to store the expansion coefficients along the analyzed path for all files analyzed.

#%% Determine the expansion in Bloch states for each of the cube files

if cube_input:
    bse_folded_states, state_nr, files_processed = bse.bse_cube(file_specifiers, log_file, k_path, kappa_path, calculation_data)
elif h5_input:
    bse_folded_states, state_nr, files_processed = bse.bse_h5(file_specifiers, log_file, k_path, kappa_path, calculation_data)


#%%

project = file_specifiers['Project']
if files_processed and np_array_dump:
    if not (re_run):
        bse.write_path(project, k_path_names, kappa_ticks, kappa_path)
    bse.write_bse_folded(project, state_nr, bse_folded_states)
    log_string = 'Bloch state expansion saved as NumPy array'
    bse.log_output(log_string, log_file)


#%% Post-process through energy binning

if files_processed and E_binning:
    if cube_input:
        E_file = file_specifiers['Project'] + '_' + file_specifiers['State'] + '-' +\
                 file_specifiers['Energy'] + '.' + file_specifiers['extension'][2]
    if h5_input:
        E_file = file_specifiers['h5_Energy'] + '.' + file_specifiers['extension'][2]
    E_list = bse.read_energy(E_file)
    State_0 = file_specifiers['cube_0']
    #bse_folded_states_binned,
    E_array, bse_folded_states_binned = bse.E_bin(bse_folded_states, E_list, State_0, dE)
    log_string = 'Energy binning completed'
    bse.log_output(log_string, log_file)

#%% Save results

if files_processed and h5_output:
    bse.hdf5_output(project, k_path_names, kappa_ticks, kappa_path, state_nr, bse_folded_states,\
                bse_folded_states_binned, E_array)
    log_string = 'Bloch state expansion saved as h5 file'
    bse.log_output(log_string, log_file)

#%% Plot one of the Bloch state expansions

if files_processed and plot:
    phi_folded_path = bse_folded_states[5]
    #Plot the results
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 1, 1)
    plt.plot(kappa_path, phi_folded_path)
    plt.xlabel('Position')
    plt.ylabel('Projected Wavefunction')
    plt.title('path 0')
    plt.xticks(kappa_ticks, k_path_names)


