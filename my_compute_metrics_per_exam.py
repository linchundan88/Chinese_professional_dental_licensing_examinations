
'''
 aa cc
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--examination_type', default='Professional_Licensing_Examination')  # Professional_Licensing_Examination  Assistant_Professional_Licensing_Examination
parser.add_argument('--instruction_no', type=int, default=0)
parser.add_argument('--thinking_suffix', default='')  # /think  /no_think
parser.add_argument('--temperature', type=int, default=1)  # 0 or 1
parser.set_defaults(unknown_as_error=False)
args = parser.parse_args()
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from libs.my_helper_exam import parse_result
from libs.my_helper_ststistics import my_bootstrap
import pickle
from collections import defaultdict


if __name__ == '__main__':
    list_options = ['A', 'B', 'C', 'D', 'E']

    list_model_name = ['qwen-max', 'qwen3-max',  'qwen-plus', 'qwen3.5-plus',  #'qwen3.5-max',
                       'gpt-3.5-turbo', 'gpt-4','gpt-5', 'gpt-5.4',
                       'doubao-seed-1.6', 'doubao-seed-1-8-251228', 'doubao-seed-2-0-pro-260215',
                       'gemini-2.5-pro',  'gemini-3-pro-preview', 'gemini-3.1-pro-preview',
                       'deepseek-v3', 'deepseek-chat']  #'deepseek-v3.2'

    list_model_name = ['gemini-2.5-pro']

    for model_name in list_model_name:
        print(f'Compute performance metrics of {model_name } temperature:{args.temperature}\n')

        filename_pkl = f'{args.examination_type}_{model_name.replace(":", "_")}_instruction_no{args.instruction_no}_{args.thinking_suffix}.pkl'
        with open(Path(__file__).resolve().parents[0] / 'results' / f'temperature_{args.temperature}' / filename_pkl, 'rb') as f:
            df, list_prediction_answer = pickle.load(f)

        num_records = len(df)
        list_correct_answers = df['答案'].tolist()
        list_exams = df['试卷编号'].tolist()

        exam_predictions = defaultdict(list)
        number_correct,  number_errors = 0, 0
        for i, (correct_answer, prediction_answer, exam_id) in enumerate(zip(list_correct_answers, list_prediction_answer, list_exams)):
            correct_answer = correct_answer.strip()
            correct_answer = correct_answer[0:1]
            prediction_answer = parse_result(prediction_answer)

            if prediction_answer in list_options:
                if correct_answer.strip() == prediction_answer.strip():
                    number_correct += 1
                    exam_predictions[exam_id].append(1)
                else:
                    number_errors += 1
                    exam_predictions[exam_id].append(0)
            else:
                print('parsing error:', prediction_answer)
                number_errors += 1
                exam_predictions[exam_id].append(0)

        # subgroup analysis for exams
        sorted_exams = sorted(exam_predictions.keys())
        exam_correct_rates = []  # 用于存储每个考试的准确率

        for exam_id in sorted_exams:
            exam_preds = exam_predictions[exam_id]
            unit_num = len(exam_preds)
            unit_correct = sum(exam_preds)
            unit_correct_rate = np.mean(exam_preds)
            exam_correct_rates.append(unit_correct_rate)  # 收集准确率

            assert unit_num>0, f'{unit_num} questions in unit {exam_id}'

            lower_95, higher_95 = my_bootstrap(exam_preds, np.mean, c=0.95)
            # lower_99, higher_99 = my_bootstrap(exam_preds, np.mean, c=0.99)

            print(f'\nExam {exam_id}:')
            # print(f'  Total questions: {unit_num}')
            # print(f'  Correct answers: {unit_correct}')
            print(f'  Correct rate: {round(unit_correct_rate,3)}')
            print(f'  95% CI: ({round(lower_95,3)} ~ {round(higher_95,3)})')
            # print(f'  99% CI: ({round(lower_99,3)} ~ {round(higher_99,3)})')

        assert len(exam_correct_rates) > 0, f'{model_name} has no exam correct answers'

        mean_accuracy = np.mean(exam_correct_rates)
        std_accuracy = np.std(exam_correct_rates, ddof=0)  # 总体标准差

        print('\n')
        print(f'Mean accuracy across all exams: {round(mean_accuracy, 3)}')
        print(f'Std dev of accuracy: {round(std_accuracy, 3)}')
        print('\n' + '=' * 60)



    print('OK')