#!/bin/bash

for instruction_no in "instruction_no0" "instruction_no1" "instruction_no2"
do
  echo "prompt using ${instruction_no}"
  for model_name in  "qwen-plus" "qwen3-max" "deepseek-v3.1" "qwen3_32b" "gpt-oss_120b"
  do
    python ./my_compute_metrics_per_question.py --filename_pkl Professional_Licensing_Examination_${model_name}_${instruction_no}_.pkl
    python ./my_compute_metrics_per_question.py --filename_pkl Assistant_Professional_Licensing_Examination_${model_name}_${instruction_no}_.pkl
  done
done

