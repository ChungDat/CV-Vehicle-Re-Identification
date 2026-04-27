#!/bin/bash

python train_model.py --model-type baseline_with_CBAM --total-epochs 60 --batch-size 32 --num-instances 8 --warmup-epochs 0 --random-erasing False --label-smoothing False "$@"