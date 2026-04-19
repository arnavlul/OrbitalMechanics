# Orbital Mechanics - Hamiltonian Neural Network (HNN)

This project implements a Hamiltonian Neural Network to predict orbital trajectories using C++ and Eigen.

## Prerequisites

- **Eigen Library**: This project uses the Eigen C++ library.
  - Download it from [eigen.tuxfamily.org](https://eigen.tuxfamily.org/).
  - Ensure the `Eigen` headers are accessible (either in an `eigen/` folder in the root or in your system include path).

## Compilation

To compile with full optimizations:

```bash
g++ -O3 hnn.cpp -o hnn.exe
```
or if you want it to faster (but machine specific):
```bash
g++ -O3 -march=native hnn.cpp -o hnn.exe
```

## Usage

1. **Prepare Data**:
   ```bash
   python clean_datafiles.py
   ```
2. **Train & Simulate**:
   ```bash
   ./hnn.exe
   ```
3. **Visualize**:
   ```bash
   python plotter.py
   ```

## Files
- `hnn.cpp`: Main HNN implementation.
- `plotter.py`: 3D orbit and loss visualization.
- `test_files/`: Source data for planetary orbits.
