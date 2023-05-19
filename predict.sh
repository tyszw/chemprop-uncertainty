#!/bin/bash

#GPU=3
DATASET_NAME=EGFR
TEST_PATH=data/own_data/${DATASET_NAME}.csv
CHECKPOINT_DIR=model/${DATASET_NAME}_checkpoints
PREDS_PATH=model/${DATASET_NAME}_checkpoints/pred_${DATASET_NAME}.csv

python predict.py \
    --test_path ${TEST_PATH} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --preds_path ${PREDS_PATH} \
    --estimate_variance
    #--gpu ${GPU} \
    #--ensemble_size ${ENSEMBLE_SIZE} \
    #--num_folds ${NUM_FOLDS} \
    #--epochs ${EPOCHS} \
    #--batch_size ${BATCH_SIZE}
