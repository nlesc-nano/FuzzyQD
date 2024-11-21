
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

### Basic Workflow

1. **Prepare an Input YAML File**: Create an `input.yaml` file with the required settings.

2. **Run the FuzzyQD Command**: Execute the `fuzzyqd` command, providing the YAML file as input:

   ```bash
   fuzzyqd input.yaml
   ```

3. **Output**: The output will include processed files, modified configurations, and optionally SLURM scripts for batch processing.

### Block Processing

If your workflow involves splitting data into blocks for parallel processing:

1. **Specify Blocks in YAML**: Add the `blocks` parameter in your `input.yaml` file.

2. **Run the FuzzyQD Command**: Execute the `fuzzyqd` command as usual. The package will automatically create folders, modify configurations, and generate SLURM scripts for each block.

### Example YAML File

Below is a sample `input.yaml` file:

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

## Contributing

Contributions to FuzzyQD are welcome! If you encounter any issues or have suggestions for new features, please open an issue or submit a pull request on [GitHub](https://github.com/nlesc-nano/fuzzyqd).

## License

FuzzyQD is licensed under the [MIT License](LICENSE.txt).
