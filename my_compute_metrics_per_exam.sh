#!/bin/bash


echo "compute metrics per exam"
for instruction_no in "instruction_no0"
do
  for model_name in  "qwen-plus" "qwen3-max" "deepseek-v3.1" "qwen3_32b" "gpt-oss_120b"
  do
#    python ./my_compute_metrics_per_exam.py --filename_pkl Professional_Licensing_Examination_${model_name}_${instruction_no}_.pkl
    python ./my_compute_metrics_per_exam.py --filename_pkl Assistant_Professional_Licensing_Examination_${model_name}_${instruction_no}_.pkl
  done

done

