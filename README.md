# Data Visualization and Statistical Analysis

This repository contains various data visualization scripts and statistical analyses related to neuronal spike rate data in PCA space.

## File Overview

- **plot_...**: Files that begin with `plot_` are dedicated to generating various data visualizations, particularly in PCA space. These visualizations explore spike rate comparisons across different sound stimuli and neuronal responses.
  
- **box_and_whiskey.py**: This is the final visualization script, presenting a summary of data through box-and-whisker plots. The script provides a comprehensive view of the variance in spike rates across different conditions.
  
- **mann_whitney_u.py**: This script performs a Mann-Whitney U test, a non-parametric statistical analysis, comparing the spike rate distributions between different conditions to test for significant differences.

## Additional Branch

- **itszoetom-patch-1**: In this branch, you'll find additional visualizations generated from the scripts mentioned above. These include detailed PCA projections, spike rate comparisons, and further explorations of the data.

## How to Use

1. Clone the repository.
2. Switch to the `itszoetom-patch-1` branch to view visualizations:
   ```bash
   git checkout itszoetom-patch-1
