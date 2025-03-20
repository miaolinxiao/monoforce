#!/bin/bash

MODEL=lss_temporal # lss lss_bevformer lss_temporal
ROBOT=marv
DEBUG=False
VIS=False
BSZ=1  # 24, 24, 4
WEIGHTS=$HOME/THESIS/monoforce/monoforce/config/weights/lss/val.pth  # config/weights/${MODEL}/val.pth

# source $HOME/workspaces/traversability_ws/devel/setup.bash
./train.py --bsz $BSZ --nepochs 100 --lr 1e-4 \
           --debug $DEBUG --vis $VIS \
           --geom_weight 1.0 --terrain_weight 3.0 --phys_weight 4.0 \
           --traj_sim_time 5.0 \
           --robot $ROBOT \
           --model $MODEL \
           --pretrained_model_path ${WEIGHTS}
