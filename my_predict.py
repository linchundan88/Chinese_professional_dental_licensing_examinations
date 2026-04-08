# sys.path.append(str(Path(__file__).resolve().parent))
import pandas as pd
from dotenv import load_dotenv
import pickle
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from libs.my_helper_exam import process_llm_prediction, list_instructions
parser = argparse.ArgumentParser()
# model_name: 'deepseek-v3.1', 'deepseek-v3', 'deepseek-r1'，'qwen-plus', 'deepseek-r1:14b' , 'deepseek-r1:32b', 'qwen3:8b', 'qwen3:14b', 'qwen3:14b', 'qwen3:32b'
# 'gpt-oss:20b' 'gpt-oss:120b'
# Qwen-plus and Deepseek-v3.1 support mixed thinking mode, and disable thinking mode by default.
# gpt-5.1-chat-latest  gemini-3-pro-preview   deepseek-chat（deepseek-v3.2)
parser.add_argument('--model_name', default='gemini-2.5-pro')
parser.add_argument('--timeout', type=int, default=120)
parser.add_argument('--examination_type', default='Professional_Licensing_Examination')  # Professional_Licensing_Examination  Assistant_Professional_Licensing_Examination
parser.add_argument('--instruction_no', type=int, default=0)
# Setting the temperature and top p to 0 should make the outputs deterministic (omit the small and uncontrollable  batch effect)
parser.add_argument('--temperature', type=int, default=0)
parser.add_argument('--max_completion_tokens', type=int, default=500)
parser.add_argument('--thinking_suffix', default='')  # /think  /no_think
parser.add_argument('--max_workers', type=int, default=1)  # the number of threads.
args = parser.parse_args()
load_dotenv()
from libs.my_helper_llm import get_llm_client



if __name__ == '__main__':
    print('model name:', args.model_name)
    print('temperature:', args.temperature)

    llm_client = get_llm_client(args.model_name,  timeout=args.timeout)

    path_save_results = Path(__file__).resolve().parents[0] / 'results' / f'temperature_{args.temperature}'
    path_save_results.mkdir(exist_ok=True, parents=True)

    input_text_prefix = f""  # 请根据题目，从以下五个选项中选择一个最正确的答案，请直接回答正确答案选项。\n

    df = pd.read_excel(Path(__file__).resolve().parent / 'datafiles' / f'Dental_{args.examination_type}.xlsx')

    list_answers = []
    start_time = time.time()

    if args.max_workers == 1:      # single thread
        for (index, record) in df.iterrows():
            answer = process_llm_prediction(record, llm_client, args.model_name, list_instructions[args.instruction_no],
                                            input_text_prefix, thinking_suffix=args.thinking_suffix,
                                            temperature= args.temperature, max_completion_tokens=args.max_completion_tokens)
            print(f'index:{index}:', answer)
            list_answers.append(answer)
    else:
        # The order of the questions has been scrambled.
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_llm_prediction,  record, llm_client, args.model_name, list_instructions[args.instruction_no],
                                       input_text_prefix, args.thinking_suffix,
                                       args.temperature, args.max_completion_tokens, None, index) for index, record in df.iterrows()]

            for future in as_completed(futures):
                index, answer = future.result()
                print(f'{index}:', answer)
                list_answers.append([index, answer])

        print('Sorting answers by index...')
        list_answers1 = sorted(list_answers, key=lambda x: x[0])
        # extract only the answers in correct order
        list_answers = [answer for index, answer in list_answers1]
        print(f'Sorted {len(list_answers)} answers')

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

Professional_Licensing_Examination_gpt-5.1-chat-latest_instruction_no0_.pkl  execution time: 5307.44 seconds



'''