# isac-mimo-ofdm-wf

## Introduction
This is the simulation code repository for the paper xxx (coming soon after accepted).

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
python ./manopt_unconstrained.py -m num_points_per_iter=1,10,30,50 multi_obj_factor=0,1e-4,2e-4,4e-4,8e-4,16e-4,32e-4,64e-4,128e-4,256e-4,512e-4,1024e-4
```
The adjustable configurations are found in `./config.yaml`.

After finishing the simulations, the results need to be post-processed for plotting. Run the script `post_process.py`. This might take a relatively long time.

The script `plotting.py` is excecuted after finishing the post-processing and should generate the same plots on our paper.
