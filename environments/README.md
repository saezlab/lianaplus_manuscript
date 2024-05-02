This directory contains the full conda environments (run on a local and cluster Ubuntu machine v 20.04 & 24.04).

`env.yml` is the general environment file that contains all the necessary packages to run the code in this repository.

`tangram_env.yml` is a specific environment file used to run deconvolution with the `tangram` package.

`mofatalk.yml` is a specific environment file that contains all the necessary packages to run the efficiency benchmark which uses the MOFA+ package for multi-view factorisation.

`spiana.yml` is a general environment file equivalent to `env.yml` used to run the efficiency benchmarks on a personal laptop.
This environment was also used to generate Figure 3M in the manuscript, due to licensing constrains on gurobi. 