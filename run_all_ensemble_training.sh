#!/bin/bash

gpu_id=0

for ds in mnist mnist-m svhn usps
do
  for i in 0 1 2 3 4 5 6 7 8 9 
  do
  CUDA_VISIBLE_DEVICES=$gpu_id python train_model.py --source-dataset $ds --model-type dann_arch --base-dir checkpoints/ensemble_dann_arch_source_models/$i/ --seed $(( 100*(i+1) ))
  done
done

for ds in mnist mnist-m svhn usps
do
  for i in 0 1 2 3 4 5 6 7 8 9 
  do
  CUDA_VISIBLE_DEVICES=$gpu_id python train_model.py --source-dataset $ds --model-type typical_dnn --base-dir checkpoints/ensemble_typical_dnn_source_models/$i/ --seed $(( 100*(i+1) ))
  done
done
