#!/bin/bash

GPU=3
DATASET_NAME=EGFR
DATA_PATH=data/own_data/${DATASET_NAME}.csv
DATASET_TYPE=regression
SEED=100

# train argments
NUM_FOLDS=5
EPOCHS=20
BATCH_SIZE=50
METRIC=r2 # {auc,prc-auc,rmse,mae,mse,r2,accuracy,cross_entropy}
ACTIVATION=ReLU # {ReLU,LeakyReLU,PReLU,tanh,SELU,ELU}

# hyperparameter optimazation argments
CONFIG_SAVE_PATH=model/${DATASET_NAME}_ckpt_${METRIC}/hyperopt.json
LOG_DIR=model/${DATASET_NAME}_ckpt_${METRIC}/hyperopt.log
NUM_ITERS=20

python hyperparameter_optimization.py \
    --data_path ${DATA_PATH} \
    --dataset_type ${DATASET_TYPE} \
    --config_save_path ${CONFIG_SAVE_PATH} \
    --log_dir ${LOG_DIR}\
    --num_iters ${NUM_ITERS}\
    --gpu ${GPU} \
    --seed ${SEED} \
    --num_folds ${NUM_FOLDS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE}
    --metric ${METRIC} \
    --activation ${ACTIVATION} \
    --save_smiles_splits \
    --aleatoric
