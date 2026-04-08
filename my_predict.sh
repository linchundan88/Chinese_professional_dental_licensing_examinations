#!/bin/bash

# bash my_predict.sh > result.txt

instruction_no="0"
temperature="0"

#for model_name in  "doubao-seed-1.6" "doubao-seed-1-8-251228" "doubao-seed-2-0-pro-260215"
#for model_name in  "gemini-2.5-pro"  "gemini-3-pro-preview" "gemini-3.1-pro-preview"   # "gemini-2.5-pro-thinking"
for model_name in  "qwen-max"  "qwen3-max" "qwen-plus"  "qwen3.5-plus"
#for model_name in  "qwen-plus"
#for model_name in  "gpt-3.5-turbo" "gpt-4" "gpt-5"  "gpt-5.4"
#for model_name in "deepseek-v3" "deepseek-chat"
 do
  echo $model_name
  python ./my_predict.py --model_name ${model_name} --examination_type Professional_Licensing_Examination --instruction_no ${instruction_no} --temperature ${temperature} --max_workers 1
done
