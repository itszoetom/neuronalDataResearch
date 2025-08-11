# Neuronal Data Research

This repository contains scripts and Jupyter notebooks for analyzing electrophysiological recordings, with a focus on evoked responses, PCA, discriminability, and ridge regression decoding. Figures are organized into subfolders for easy retrieval.

## Repository Structure

### **Evoked**
- **Discriminability**: Uses d-prime and LDA to measure neural discrimination of stimuli.
- **PCA**: Principal Component Analysis of evoked neural activity with statistical tests (Mann–Whitney U, t-tests).
- **Ridge**: Decoding analysis using ridge regression, including full dataset, subsets, and more complex models.

### **Spike Windows**
- Repeated evoked period analysis for three defined spike windows: onset, sustained, and offset.

### **Single Mouse Analysis**
- Scripts and notebooks for analyses at the individual mouse level.

### **Figures**
- Organized into subfolders for population-level, single-mouse, ridge regression, and poster-ready plots.

### **Supporting Scripts**
- **database_generation.py**: Builds experimental database.
- **funcs.py**: Shared helper functions.
- **neuropix_split_multisession.py**: Processes multi-session Neuropixels data.
- **params.py**: Analysis parameters.
