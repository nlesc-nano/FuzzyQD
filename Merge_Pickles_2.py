# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:48:06 2024

@author: zhens
"""

""" Program used to merge multiple files containing NumPy arrays saved as *.pkl and calculate an output map of BSE 
versus energy """

import sys
sys.path.append('/Users/zhens/Documents/Sabbatical/BandStructure/AnalysisPrograms')

import pickle
import os
import numpy as np
import bse_def_3 as bse


#%%Conversion configuration

hartree = 27.2107
dE = 0.0125 / hartree

pkl = True
pdos = True

State_min = 7964
pkl_step = 100
pkl_N = 18
pdos_number = np.array([1, 2, 3])
element_N = pdos_number.size


#%%Folder and file configuration

folder = '/Users/zhens/Documents/Sabbatical/BandStructure/CsPbBr3/tetra/HLE17/'
in_out_data = {'Project': 'InP_jordi', 'input_pkl': '_bse_States', 'input_pdos': '_CUBES', 'Output_pdos': '_pdos_binned'}
extensions = ['pkl', 'pdos']
os.chdir(folder)

pdos_files = [''] * element_N
element_list = [''] * element_N
pkl_files = [''] * pkl_N

#bse_files = {'File1': 'InP_608_bse_States_2382_2481.pkl', 'File2': 'InP_608_bse_States_2482_2881.pkl',\
#             'File3': 'InP_608_bse_States_2882_2981.pkl'} 

if (pdos):
    for element in range(element_N):
        file = in_out_data['Project'] + in_out_data['input_pdos'] + '-k' + str(pdos_number[element]) + '-1.' + extensions[1]
        pdos_files[element] = file
        element_list[element] = bse.readatom(file)
    energies = np.loadtxt(pdos_files[0], usecols=(0,1))
    # energies[:, 1] *= 27.211
    # Read Files with PDOS info
    xs = [np.loadtxt(pdos_files[i]) for i in range(element_N)]
    # Add up all orbitals contribution for each atom type
    ys = np.stack([np.sum(xs[i][:, 3:], axis=1) for i in range(len(pdos_files))]).transpose()
    energies_0 = int(energies[0, 0])
    clip_min, clip_max = State_min - energies_0, State_min - energies_0 + pkl_step * pkl_N 
    ys_clipped = ys[int(clip_min) : int(clip_max), :]
    pdos_E_array, pdos_binned = bse.E_bin(ys_clipped, energies, State_min, dE)
    pdos_E_array -= dE / 2
    pdos_E_array = pdos_E_array[1:]
    
#%%

if (pkl):
    state_nr = np.array([])
    bse_map_load = []
    for pkl_i in range (pkl_N):
        State_i = State_min + pkl_step * pkl_i
        State_f = State_i + pkl_step - 1
        file = in_out_data['Project'] + in_out_data['input_pkl'] + '_' + str(State_i) + '_' + str(State_f) + '.' + extensions[0]
        pkl_files[pkl_i] = file
        with open(file, 'rb') as f:
            arrays = pickle.load(f)
            array1, array2 = arrays
            state_nr = np.hstack((state_nr, array1))
            bse_map_load.append(array2)
        bse_map = np.vstack(bse_map_load)
    if (not(pdos)):
        energy_file = in_out_data['Project'] + '_STATES-E.txt'
        energies = bse.read_energy(energy_file)
    k_path_file = in_out_data['Project'] + '_bse_k_path.pkl'    
    with open(k_path_file, 'rb') as f:
        arrays = pickle.load(f)
        k_path_names, kappa_ticks, kappa = arrays

#%%

#Read energy list and perform energy binning on merged array

bse_E_array, bse_map_binned = bse.E_bin(bse_map, energies, State_min, dE)
#pdos_E_array, pdos_binned = bse.E_bin(bse_map, energies, State_min, dE)

#%%

#Output to h5 file
project = in_out_data['Project']

if (pkl):
    bse.hdf5_output(project, k_path_names, kappa_ticks, kappa, state_nr, bse_map.T, bse_map_binned.T, bse_E_array * hartree)

if (pdos):
    bse.hdf5_pdos_out(project, State_min, ys_clipped, pdos_binned, pdos_E_array * hartree, element_list)




# Process the arrays as needed
#print("Array 1 data:")
#print(array1)
#print("Array 2 data:")
#print(array2)
#print("Array 3 data:")
#print(array3)
    
