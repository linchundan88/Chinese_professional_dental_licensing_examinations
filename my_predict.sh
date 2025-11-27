#!/bin/bash

# bash my_predict.sh > result.txt

instruction_no="2"

for model_name in  "qwen-plus" "qwen3-max" "deepseek-v3.1" "qwen3:32b" "gpt-oss:120b" "gpt-oss:20b"
#for model_name in  "qwen-plus"
do
  python ./my_predict.py --model_name ${model_name} --examination_type Professional_Licensing_Examination --instruction_no ${instruction_no}
  python ./my_predict.py --model_name ${model_name} --examination_type Assistant_Professional_Licensing_Examination --instruction_no ${instruction_no}
done
