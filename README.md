# isac-mimo-ofdm-wf

## Introduction
This is the simulation code repository for the paper **Optimal Linear Precoder Design for MIMO-OFDM Integrated Sensing and Communications Based on Bayesian Cramér-Rao Bound**, already accepted by [IEEE Globecom 2023](https://globecom2023.ieee-globecom.org/).

## How To Use
To reproduce the results in the paper, first create a conda environment
```
conda create -n isac-mimo-ofdm python=3.10 pip
```

Activate the created env and install the required python packages
```
conda activate isac-mimo-ofdm
pip install -r ./requirements.txt
```

The main script is `./manopt_unconstrained.py`. Multi-configuration simualtion is enabled by the `multirun` machanism of [hydra](https://hydra.cc/). For example, to run the simualtion with the same configurations as our paper, run the following command
```
python ./manopt_unconstrained.py -m num_points_per_iter=1,10,50,100 multi_obj_factor=multi_obj_factor='range(0,1.01,0.1)'
```
The adjustable configurations are found in `./config.yaml`.

After finishing the simulations, the results need to be post-processed for plotting. Run the script `post_process.py`. This might take a relatively long time.

The script `plotting.py` is excecuted after finishing the post-processing and should generate the same plots on our paper.
