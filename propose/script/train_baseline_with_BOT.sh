#!/bin/bash

python train_model.py --model-type baseline_with_BOT --total-epochs 60 --batch-size 32 --num-instances 8 --warmup-epochs 10 --random-erasing True --label-smoothing True "$@"