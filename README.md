# Choquet-Integral 
# SynthNonLinearClassifiedDataGen: Non-Linear Data Classification Using Genetic Algorithm

## Overview
This project implements a synthetic non-linear classified data generator and classifier using genetic algorithms. The implementation is based on a [research paper](https://github.com/CasterShade/Choquet-Integral/blob/main/Nonlinear_Classification_by_Genetic_Algorithm_with_Signed_Fuzzy_Measure%20(6).pdf) and includes functionalities for generating synthetic data, classifying it using Choquet hyperplanes, and optimizing the classification process through a genetic algorithm.

The solution supports **N-dimensional data classification** and visualizes results for 2D cases. The attached Python file contains the full implementation.

## Results

### Target Classification
The following graph shows the target classification of synthetic data:
![Target Classification](https://github.com/CasterShade/Choquet-Integral/blob/main/image%20(1).png)

### Genetic Algorithm Optimization
Using the genetic algorithm approach described in the paper, the generated result is shown below:
![Genetic Algorithm Result](https://github.com/CasterShade/Choquet-Integral/blob/main/image%20(2).png)

The printed number on top of each graph is the **chromosome** used to create the **Choquet hyperplane** for classification:
![Chromosome-Based Classification](https://github.com/CasterShade/Choquet-Integral/blob/main/image%20(2).png)

### Convergence Details
- **Number of generations:** 45
- **Precision:** \(10^{-3}\)
- **Consecutive difference between max distances:** \(10^{-4}\)

The genetic algorithm converged successfully within the specified thresholds.

---

## Features

1. **Synthetic Data Generation**:
   - Generates \(N\)-dimensional data points.
   - Supports random and user-defined intersection points.
   - Random hyperplane generation for non-linear classification.

2. **Non-Linear Classification**:
   - Classifies data points using hyperplanes or Choquet hyperplanes.
   - Choquet integral-based classification for complex decision boundaries.

3. **Genetic Algorithm**:
   - Implements fitness calculation, crossover, mutation, and parent selection.
   - Optimizes the Choquet hyperplane parameters for classification.
   - Adjustable parameters for precision, mutation rate, and population size.

4. **Visualization**:
   - Displays classified data with decision boundaries for 2D cases.

---

## Usage

### Prerequisites
- Python 3.6 or later
- Required libraries: `numpy`, `matplotlib`

### How to Run
1. Clone the repository.
2. Run the provided script:
   ```bash
   python NonLinearClassifiedSyntheticNDimDataGenV2.py
   ```

3. Customize the parameters for dimensions, data size, and genetic algorithm settings within the script:
   ```python
   dg = SynthNonLinearClassifiedDataGen(
       dimensions=2,
       size=200,
       intersection=[0.2, 0.5],
       intersection_flag=False,
       planes=initialize_plane,
       chromosome="001000100000110..."
   )
   ```

4. Visualize results using:
   ```python
   dg.visualize(data)
   dg.visualize_choquet(data, "chromosome_here")
   ```

5. Run the genetic algorithm:
   ```python
   best_chromosome, cur_D, generation_count = dg.learn()
   print(f"Best Chromosome: {best_chromosome}")
   ```

---

## Known Issues
- **Performance**: The code is currently slower for higher dimensions. Optimization is planned.
- **Visualization**: Only supports 2D visualization for now.

---

## Future Enhancements
- Improve computational efficiency for higher dimensions.
- Extend visualization support for 3D and higher-dimensional data projections.
- Experiment with additional optimization techniques to compare with genetic algorithms.

---

## Contribution
I am open to suggestions, improvements, and collaborative experiments. Feel free to fork the repository or raise issues for discussion.

---

## Contact
**Author**: Mohammad Zubair Khan  
Feel free to reach out with questions or ideas for enhancement.
