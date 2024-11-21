
# FuzzyQD

**FuzzyQD** is a Python package for Bloch state expansion and quantum dot fuzzy band structures. It facilitates the analysis and computation of band structures in quantum dots, enabling efficient and structured workflows for scientific research.

## Features

- **Bloch State Expansion (BSE)**: Analyze and expand Bloch states for quantum dots.
- **Flexible Input Handling**: Supports cube and HDF5 files for input.
- **Parallel Processing**: Includes support for multi-node batch processing using SLURM scripts.
- **Customizable Settings**: Configure lattice parameters, reciprocal space details, and clipping options through an easy-to-use YAML configuration file.

## Requirements

FuzzyQD requires Python 3.8 or higher and the following dependencies:

- `numpy`
- `matplotlib`
- `h5py`
- `PyYAML`
- `scipy`
- `joblib`

## Installation

You can install the FuzzyQD package directly from the source:

### Clone the Repository

```bash
git clone https://github.com/nlesc-nano/fuzzyqd.git
cd fuzzyqd
```

### Install the Package

To install the package, use `pip`:

```bash
pip install .
```

For development purposes, you can install it in editable mode:

```bash
pip install -e .
```

### Verify the Installation

Run the following command to verify the installation:

```bash
fuzzyqd --help
```

If the installation was successful, this will display the usage information for the `fuzzyqd` command.

## Usage

Follow these instructions to use the **FuzzyQD** package effectively:

---

### **1. Prepare Cube Files**
1. Run a **geometry optimization** for your quantum dot (QD) using the latest version of **Cp2k**.
   - We assume you're using **Density Functional Theory (DFT)**.
2. After the geometry optimization, perform a **single-point calculation** on the optimized structure with the following settings in the `&DFT` block of your input file:

   ```plaintext
   &PRINT
     &PDOS
       LOG_PRINT_KEY
       NLUMO 1000
       &EACH
         GEO_OPT 500
       &END EACH
       ADD_LAST NUMERIC
     &END PDOS
     &MO_CUBES
       LOG_PRINT_KEY
       NHOMO 150
       NLUMO 150
       WRITE_CUBE .TRUE.
       &EACH
         GEO_OPT 500
       &END EACH
       ADD_LAST NUMERIC
     &END MO_CUBES
   &END PRINT
   ```

   - This configuration writes the **highest 150 CUBEs for the HOMOs** and the **lowest 150 CUBEs for the LUMOs**.
   - It also outputs PDOS (Projected Density of States) files.

---

### **2. Set Up the YAML Configuration**
1. Move to the folder containing the cube files.
2. Copy the example `input.yaml` file (found in the `example/` directory of this repository) into this folder.
3. Edit the `input.yaml` file to match your system settings:
   - Update the **lattice constant** and **k_path points** for the band structure calculation.
   - Set the `Project` name to match the `PROJECT` name defined in your Cp2k input file:

     ```plaintext
     &GLOBAL
       PROJECT GaAs
       RUN_TYPE ENERGY
       PRINT_LEVEL LOW
     &END GLOBAL
     ```

   - Specify the **number of blocks** to divide the cube files for parallel execution on HPC infrastructure.
   - Define the **first cube file number**.

4. A typical `input.yaml` file might look like this:

   ```yaml
   lattice:
     a: 5.5973
   clipping:
     clip: false
   reciprocal_space:
     dk: 0.005
   energy_binning:
     dE: 0.0125
   files:
     folder_cubes: ./cubes
     Project: GaAs
     cube_0: 1086
     block_index: 1
     N_cube: 25
   settings:
     fcc: true
     cube_input: true
   k_path:
     names: ['L', 'G', 'Xx', 'W', 'K', 'G', 'Xy']
     points:
       - [0.25, 0.25, 0.25]
       - [0.0, 0.0, 0.0]
       - [0.5, 0.0, 0.0]
       - [0.5, 0.25, 0.0]
       - [0.375, 0.375, 0.0]
       - [0.0, 0.0, 0.0]
       - [0.0, 0.5, 0.0]
   ```

---

### **3. Run the Script**
Execute the following command to divide the cube files into blocks for parallel processing:

```bash
fuzzyqd input.yaml
```

---

### **4. Adjust SLURM Files**
After running the command, several block folders will be created. Each folder will contain:
- A chunk of cube files.
- A `slurm` input script (`run_bse.slurm`).
- An updated `input_parameters.yaml`.

Update the `slurm` script in each folder according to your HPC infrastructure and personal preferences.

---

### **5. Submit Jobs on HPC**
Submit the jobs for each block folder to your HPC system using:

```bash
sbatch run_bse.slurm
```

---

### **6. Process Locally (Optional)**
If you're running the calculations on a personal computer instead of an HPC system, you can process each block folder locally:

```bash
fuzzyqd input_parameters.yaml
```

---

### **7. Merge Results**
Once the cube files in each block folder are processed, two `.pkl` files will be generated in each folder. To merge these results:

1. Create a new folder (e.g., `process_pickles`):
   ```bash
   mkdir process_pickles
   cd process_pickles
   ```

2. Copy all `.pkl` files from each block folder into the `process_pickles` folder:
   ```bash
   cp ../block_folder_*/GaAs_*.pkl .
   ```

3. Additionally, copy all `*.pdos` files generated in the original Cp2k calculations into the `process_pickles` folder:
   ```bash
   cp ../path_to_cp2k_calculations/*.pdos .
   ```

4. Run the following command to process the pickles and the `*.pdos` files, and generate a combined HDF5 file:
   ```bash
   process_pickles.py --bse --folder . --project GaAs
   ```

This will generate a combined HDF5 file containing the processed results and the relevant PDOS information.

---

### **8. Plot the Fuzzy Band Structure**
To visualize the fuzzy band structure:
1. Run the following command:
   ```bash
   plot_fuzzyqd.py --hdf5 nameoffile.h5
   ```
2. To plot the band structure in a specific energy range (in eV), use:
   ```bash
   plot_fuzzyqd.py --hdf5 nameoffile.h5 --energy_window -3.0 3.0
   ```

---

### Notes
- Replace `nameoffile.h5` with the name of your HDF5 file.
- The energy range in the `--energy_window` flag is specified in electron volts (eV).

## Contributing

Contributions to FuzzyQD are welcome! If you encounter any issues or have suggestions for new features, please open an issue or submit a pull request on [GitHub](https://github.com/nlesc-nano/fuzzyqd).

## License

FuzzyQD is licensed under the [MIT License](LICENSE.txt).
