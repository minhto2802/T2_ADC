#!/usr/bin/env bash
run_id=$1
port_id=$2
cd F:/BACKUPS/Pix2Pix_T22ADC_cancer_classification/run$run_id/
tensorboard --logdir=./logs --host=127.0.0.1 --port=$port_id
