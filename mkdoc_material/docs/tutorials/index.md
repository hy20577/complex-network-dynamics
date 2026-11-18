# Guides & Tutorials

Each topic below comes in two forms:

- a **guide**: this website's page, with the explanation, the figures and the results, to read;
- a **tutorial**: the Jupyter notebook behind the page, with all the code, to run step by step.

!!! tip "Run the tutorials yourself"

    Every tutorial is shared as a Jupyter notebook. The **Code** line at the top of each page lets you:

    - :material-google: **Open in Colab**: run the notebook in your browser, with nothing to install;
    - :material-download: **Download the notebook**: run it on your own computer with Jupyter;
    - :fontawesome-brands-github: **View on GitHub**: read the code online.

The topics go from basic network theory to inferring networks from data.

## Network structure

1. **[Introduction to Network Theory](01_introduction_to_network_theory.ipynb)**: graphs, adjacency matrices and basic measures.
2. **[Network Types](02_network_types.ipynb)**: random, small-world and scale-free networks and how they compare.

## Dynamical systems

3. **[Discrete Maps: from One Map to a Network](03_discrete_maps.ipynb)**: the logistic and circle maps, their bifurcation diagrams and Lyapunov exponents, and maps coupled on the 12- and 16-node networks.
4. **[Integrating ODEs: Euler, Runge-Kutta and Lyapunov Exponents](04_integrating_odes.ipynb)**: the Euler, RK4 and adaptive RK45 methods compared on the Lotka-Volterra and Lorenz systems, and Lorenz systems coupled on a network.

## Network inference

5. **[Correlation and Mutual Information](05_correlation_and_mutual_information.ipynb)**: Pearson correlation versus mutual information for detecting links.
6. **[Network Inference with MIR and APMSD](06_apmsd_network_inference.ipynb)**: the full method of the paper on its 4-node example, compared with the published results.
7. **[Phase and Amplitude Coupling in APMSD](07_phase_and_amplitude_coupling.ipynb)**: a toy model whose links are carried by the phase, the amplitude or both, and where in the (pc<sub>1</sub>, pc<sub>2</sub>) plane APMSD detects each of them.
