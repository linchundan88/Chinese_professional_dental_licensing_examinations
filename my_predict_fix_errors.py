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
parser.add_argument('--examination_type', default='Professional_Licensing_Examination')  # Professional_Licensing_Examination  Assistant_Professional_Licensing_Examination
parser.add_argument('--instruction_no', type=int, default=0)
parser.add_argument('--thinking_suffix', default='')  # /think  /no_think
parser.add_argument('--temperature', type=int, default=0)
parser.add_argument('--timeout', type=int, default=300)
args = parser.parse_args()
load_dotenv()
from libs.my_helper_llm import get_llm_client



if __name__ == '__main__':

    list_model_name = ['gemini-2.5-pro']
    for model_name in list_model_name:

        llm_client = get_llm_client(model_name, timeout=args.timeout)

        path_save_results = Path(__file__).resolve().parents[0] / 'results' / f'temperature_{args.temperature}'
        path_save_results.mkdir(exist_ok=True, parents=True)

        input_text_prefix = f""  # 请根据题目，从以下五个选项中选择一个最正确的答案，请直接回答正确答案选项。\n

        file_pkl = path_save_results / f'{args.examination_type}_{model_name.replace(":", "_")}_instruction_no{args.instruction_no}_{args.thinking_suffix}.pkl'
        with open(Path(__file__).resolve().parents[0] / 'results' / file_pkl, 'rb') as f:
            df, list_prediction_answer = pickle.load(f)

        list_answers = []

        for (index, record) in df.iterrows():
            current_answer = list_prediction_answer[index]
            # print(list_prediction_answer[index])
            if ('System Error: Request timed out.' in list_prediction_answer[index]
                    or '502 Bad Gateway' in list_prediction_answer[index] ):
                print(f'{index}:')
                answer = process_llm_prediction(record, llm_client, model_name, list_instructions[args.instruction_no],
                                                input_text_prefix, thinking_suffix=args.thinking_suffix, temperature=args.temperature)
                print(f'{index}:', answer)
                list_answers.append(answer)
            else:
                list_answers.append(list_prediction_answer[index])


        print(f'save to pkl file: {file_pkl}')
        pickle.dump((df, list_answers), open(file_pkl, 'wb'))

    print('OK.')
