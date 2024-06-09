#!/bin/bash

###############################################
# Usage:
#
#  ./src/reference_implementations/fairness_measurement/czarnowska_analysis/launch_cot_prompt_experiments.sh \
#   run_id \
#   dataset
# Example
#  ./src/reference_implementations/fairness_measurement/czarnowska_analysis/launch_cot_prompt_experiments.sh \
#   "run_1" \
#   "SST5"
###############################################

RUN_ID=$1
DATASET=$2

# Run the ID and DATASET for each of the llama, and llama2 models

# LLaMA
SBATCH_COMMAND="src/reference_implementations/fairness_measurement/czarnowska_analysis/run_llama_experiment_cot.slrm \
    ${RUN_ID} \
    ${DATASET}"
echo "Running sbatch command ${SBATCH_COMMAND}"
sbatch ${SBATCH_COMMAND}

# Llama 2
SBATCH_COMMAND="src/reference_implementations/fairness_measurement/czarnowska_analysis/run_llama2_experiment_cot.slrm \
    ${RUN_ID} \
    ${DATASET}"
echo "Running sbatch command ${SBATCH_COMMAND}"
sbatch ${SBATCH_COMMAND}

echo Experiments Launched
