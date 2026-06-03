#!/bin/bash

python train_model.py --model-type BOT_with_CBAM --total-epochs 60 --batch-size 32 --num-instances 8 --warmup-epochs 10 --random-erasing True --label-smoothing True --distance-mode cosine"$@"