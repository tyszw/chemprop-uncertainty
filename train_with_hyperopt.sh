#!/bin/bash

GPU=3
DATASET_NAME=EGFR

# training atguments
NUM_FOLDS=5
EPOCHS=30
BATCH_SIZE=50
METRIC=rmse # {auc,prc-auc,rmse,mae,mse,r2,accuracy,cross_entropy}
ACTIVATION=ReLU # {ReLU,LeakyReLU,PReLU,tanh,SELU,ELU}
SEED=100


DATA_PATH=data/own_data/${DATASET_NAME}.csv
DATASET_TYPE=regression

CONFIG_PATH=model/${DATASET_NAME}_ckpt_${METRIC}/hyperopt/hyperopt.json
SAVE_DIR=model/${DATASET_NAME}_ckpt_${METRIC}/hyperopt


# model
ENSEMBLE_SIZE=5

python train.py \
    --data_path ${DATA_PATH} \
    --dataset_type ${DATASET_TYPE} \
    --save_dir ${SAVE_DIR} \
    --gpu ${GPU} \
    --seed ${SEED}  \
    --ensemble_size ${ENSEMBLE_SIZE} \
    --num_folds ${NUM_FOLDS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --metric ${METRIC} \
    --activation ${ACTIVATION} \
    --save_smiles_splits \
    --aleatoric \
    --config_path ${CONFIG_PATH} \
    #--crossval_index_dir ${CROSSVAL_INDEX_DIR} \
    #--crossval_index_file ${CROSSVAL_INDEX_FILE} \
