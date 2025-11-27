
import sys
from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parent))
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import time
from pathlib import Path
parser = argparse.ArgumentParser()
# model_name: 'deepseek-v3.1', 'deepseek-v3', 'deepseek-r1'，'qwen-plus', 'deepseek-r1:14b' , 'deepseek-r1:32b', 'qwen3:8b', 'qwen3:14b', 'qwen3:14b', 'qwen3:32b'
# 'gpt-oss:20b' 'gpt-oss:120b'
# Qwen-plus and Deepseek-v3.1 support mixed thinking mode, and disable thinking mode by default.
parser.add_argument('--model_name', default='gpt-oss:120b')  # deepseek-v3.1 qwen-plus qwen3:32b
parser.add_argument('--examination_type', default='Professional_Licensing_Examination')  # Professional_Licensing_Examination  Assistant_Professional_Licensing_Examination
parser.add_argument('--instruction_no', type=int, default=1)
parser.add_argument('--thinking_suffix', default='')  # /think  /no_think
parser.add_argument('--max_workers', type=int, default=1)  # the number of threads.
args = parser.parse_args()
load_dotenv()


def process_llm_prediction(record1, chat_client, model_name, str_instruction, input_text_prefix, thinking_suffix='', list_options=None):
    if list_options is None:
        list_options = ['A', 'B', 'C', 'D', 'E']

    input_text = input_text_prefix

    input_text += record1['题干'].replace(' ', '') + ':' + '\n'
    for option in list_options:
        input_text += f'选项{option}：' + str(record1[f"选项{option}"]).replace(' ', '') + '\n'

    if thinking_suffix != '':
        input_text += thinking_suffix

    try:
        completion = chat_client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'system', 'content': str_instruction},
                      {'role': 'user', 'content': input_text}],
            # extra_body={"enable_thinking": False}   # useless
            # https://platform.openai.com/docs/api-reference/completions/create#chat-create-stop
            # A Temperature of 0, a Top K of 1, or a Top P of 0 is the same as replacing softmax with the argmax formula.
            # temperature=1,  # it is a scaling factor applied to the log-probs (or logits) prior to the softmax application.
            # A high value for Temperature squeezes all of our options closer to each other, so they have a closer probability, or a small value for Temperature stretches them apart, making the probabilities of each option further apart.
            # top_p=1,  # default value of 1
        )
    except Exception as e:
        error_msg = f"System Error: {e}"
        return error_msg

    prediction = completion.choices[0].message.content

    return prediction


if __name__ == '__main__':
    if args.model_name in ['deepseek-v3', 'deepseek-v3.1', 'deepseek-r1', 'qwen-plus', 'qwen3-max']:
        client = OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("API_KEY_ALI"),
        )
    else:
        client = OpenAI(
            base_url=os.getenv("LOCAL_SERVICE_URL"),
            api_key='ollama',  # required, but unused
        )

    path_save_results = Path(__file__).resolve().parents[0] / 'results'
    path_save_results.mkdir(exist_ok=True, parents=True)

    list_instructions = ["您是一个资深的口腔医生，请根据问题描述从给出的五个选项中选择一个最正确的答案，请直接回答该正确答案的编号。",
                         "您是一个资深的口腔医生，下面问题所给出的五个选项中只有一个是最正确的，如果您比较确定那个选项是正确的请直接回答该选项的编号，否则回答我不知道。",
                         "您是一个资深的口腔医生，下面问题所给出的五个选项中只有一个是最正确的。 如果您有90%以上的把握度能够正确回答该问题则请直接回答该选项的编号，否则回答我不知道。如果回答正确得1分，回答我不知道得0分，回答错误得负1分"
                         ]

    input_text_prefix = f""  # 请根据题目，从以下五个选项中选择一个最正确的答案，请直接回答正确答案选项。\n

    df = pd.read_excel(Path(__file__).resolve().parent / 'datafiles' / f'Dental_{args.examination_type}.xlsx')

    list_answers = []

    # with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    #     results = [None] * len(df)  # Pre-allocation result list
    #
    #     futures = [executor.submit(process_llm_prediction, record, client, args.model_name, list_instructions[args.instruction_no],
    #       input_text_prefix, thinking_suffix=args.thinking_suffix)
    #       for (index, record) in df.iterrows()]
    #     for future in as_completed(futures):
    #         answer = future.result()
    #         print(answer)
    #         list_answers.append(answer)

    # single thread
    start_time = time.time()
    for (index, record) in df.iterrows():
        answer = process_llm_prediction(record, client, args.model_name, list_instructions[args.instruction_no],
                                        input_text_prefix, thinking_suffix=args.thinking_suffix)
        # print(f'index:{index}, prediction:{answer}')
        print(f'index:{index}')
        list_answers.append(answer)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution time: {execution_time:.2f} seconds")

    file_pkl = path_save_results / f'{args.examination_type}_{args.model_name.replace(":", "_")}_instruction_no{args.instruction_no}_{args.thinking_suffix}.pkl'
    print(f'save to pkl file: {file_pkl}')
    pickle.dump((df, list_answers), open(file_pkl, 'wb'))

    print('OK.')

'''
Professional_Licensing_Examination number of records: 2400
Assistant_Professional_Licensing_Examination number of records: 1498

Professional_Licensing_Examination_qwen-plus_instruction_no1_.pkl  execution time: 1120.26 seconds   1120.26  / 2400 = 0.466775
Assistant_Professional_Licensing_Examination_qwen-plus_instruction_no1_.pkl  execution time: 1393.17 seconds   1393.17  / 1498 = 0.93

Professional_Licensing_Examination_qwen3-max_instruction_no1_.pkl  execution time: 1478.31 seconds    1478.31 / 2400  = 0.6159625
Assistant_Professional_Licensing_Examination_qwen3-max_instruction_no1_.pkl  execution time: 983.09 seconds  983.09  / 1498 = 0.656268

Professional_Licensing_Examination_deepseek-v3.1_instruction_no1_.pkl  execution time: 4142.36 seconds    4142.36 / 2400  = 1.72598 
Assistant_Professional_Licensing_Examination_deepseek-v3.1_instruction_no1_.pkl  execution time: 1205.80 seconds  1205.80  / 1498 = 0.8049

Professional_Licensing_Examination_qwen3_32b_instruction_no1_.pkl  execution time: 46259.44 seconds    46259.44 / 2400  = 19.27476
Assistant_Professional_Licensing_Examination_qwen3_32b_instruction_no1_.pkl  execution time: 29836.19 seconds   29836.19  / 1498 = 19.917349

Professional_Licensing_Examination_gpt-oss_120b_instruction_no1_.pkl  execution time: 11502.46 seconds  11502.46 / 2400  = 4.79269
Assistant_Professional_Licensing_Examination_gpt-oss_20b_instruction_no1_.pkl execution time: 8285.57 seconds    8285.57  / 1498 = 5.531

'''