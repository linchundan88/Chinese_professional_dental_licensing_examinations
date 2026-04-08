
'''

'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--examination_type', default='Professional_Licensing_Examination')  # Professional_Licensing_Examination  Assistant_Professional_Licensing_Examination
parser.add_argument('--instruction_no', type=int, default=0)
parser.add_argument('--thinking_suffix', default='')  # /think  /no_think
parser.add_argument('--temperature', type=int, default=0)  # 0 or 1
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
import random
from collections import defaultdict

if __name__ == '__main__':
    list_options = ['A', 'B', 'C', 'D', 'E']

    random.seed(800)
    np.random.seed(800)

    list_model_name = ['qwen3.5-plus', 'qwen-max', 'qwen3-max',   #'qwen3.5-max', 'qwen-plus', 'qwen3.5-plus'
                       'gpt-3.5-turbo', 'gpt-4', 'gpt-5', 'gpt-5.4',
                       'doubao-seed-1.6', 'doubao-seed-1-8-251228', 'doubao-seed-2-0-pro-260215',
                       'gemini-2.5-pro', 'gemini-3-pro-preview', 'gemini-3.1-pro-preview',
                       'deepseek-v3', 'deepseek-chat' ] #'deepseek-v3.2'

    list_model_name = ['gemini-2.5-pro']

    for model_name in list_model_name:
        print(f'Compute performance metrics of {model_name }\n')

        filename_pkl = f'{args.examination_type}_{model_name.replace(":", "_")}_instruction_no{args.instruction_no}_{args.thinking_suffix}.pkl'
        with open(Path(__file__).resolve().parents[0] / 'results' / f'temperature_{args.temperature}' / filename_pkl, 'rb') as f:
            df, list_prediction_answer = pickle.load(f)

        num_records = len(df)
        list_correct_answers = df['答案'].tolist()
        list_units = df['单元编号'].tolist()

        list_predictions = []
        number_unknown, number_parsing_errors = 0, 0
        number_correct, number_errors = 0, 0
        unit_predictions = defaultdict(list)

        for i, (correct_answer, prediction_answer, unit_id) in enumerate(zip(list_correct_answers, list_prediction_answer, list_units)):
            correct_answer = correct_answer.strip()
            correct_answer = correct_answer[0:1]
            prediction_answer = parse_result(prediction_answer)

            if prediction_answer in ['我不知道']:
                number_unknown += 1
                unit_predictions[unit_id].append(0)
            elif prediction_answer in list_options:
                if correct_answer.strip() == prediction_answer.strip():
                    number_correct += 1
                    list_predictions.append(1)
                    unit_predictions[unit_id].append(1)
                else:
                    number_errors += 1
                    list_predictions.append(0)
                    unit_predictions[unit_id].append(0)
            else:
                print('parsing error:', prediction_answer)
                number_parsing_errors += 1
                list_predictions.append(0)
                unit_predictions[unit_id].append(0)

        if number_parsing_errors > 0:
            print(f'number of parsing errors: {number_parsing_errors}, percent of parsing errors: {number_parsing_errors / num_records:.3%}')
        if number_unknown > 0:
            print(f'number of unknown: {number_unknown}, percent of error unknown: {round(number_unknown / num_records,3)}')
        # print(f'number of correct answers: {number_correct}, percent of correct answers: {round(number_correct / num_records,3)}')
        # print(f'number of error answers: {number_errors}, percent of error answers: {round(number_errors / num_records,3)}')

        correct_rate = np.mean(list_predictions)
        lower_correct_rate_95, higher_correct_rate_95 = my_bootstrap(list_predictions, np.mean, c=0.95)
        lower_correct_rate_99, higher_correct_rate_99 = my_bootstrap(list_predictions, np.mean, c=0.99)
        print(f'correct_rate:{round(correct_rate,3)}')
        print(f'(lower_correct_rate_95:{round(lower_correct_rate_95,3)}~higher_correct_rate_95:{round(higher_correct_rate_95,3)})')
        # print(f'(lower_correct_rate_99:{round(lower_correct_rate_99,3)}~higher_correct_rate_99:{round(higher_correct_rate_99,3)})')

        if number_unknown > 0:
            list_unknown = [0 for _ in range(number_unknown)]
            list_predictions.extend(list_unknown)

            correct_rate_1 = np.mean(list_predictions)
            lower_correct_rate_95_1, higher_correct_rate_95_1 = my_bootstrap(list_predictions, np.mean, c=0.95)
            lower_correct_rate_99_1, higher_correct_rate_99_1 = my_bootstrap(list_predictions, np.mean, c=0.99)

            print(f'correct_rate_1:{correct_rate_1:.4f}')
            print(f'(lower_correct_rate_95_1:{round(lower_correct_rate_95_1,3)}~higher_correct_rate_95:{round(higher_correct_rate_95_1,3)})')
            # print(f'(lower_correct_rate_99_1:{round(lower_correct_rate_99_1,3)}~higher_correct_rate_99:{round(higher_correct_rate_99_1,3)})')

        print('\n' + '='*60)
        print('Performance metrics by unit:')

        # subgroup analysis for units
        sorted_units = sorted(unit_predictions.keys())
        unit_correct_rates = []  # 用于存储每个单元的准确率

        for unit_id in sorted_units:
            unit_preds = unit_predictions[unit_id]
            unit_num = len(unit_preds)
            unit_correct = sum(unit_preds)
            unit_correct_rate = np.mean(unit_preds)
            unit_correct_rates.append(unit_correct_rate)  # 收集准确率

            assert unit_num>0, f'{unit_num} questions in unit {unit_id}'

            lower_95, higher_95 = my_bootstrap(unit_preds, np.mean, c=0.95)
            lower_99, higher_99 = my_bootstrap(unit_preds, np.mean, c=0.99)

            print(f'\nUnit {unit_id}:')
            print(f'  Total questions: {unit_num}')
            print(f'  Correct answers: {unit_correct}')
            print(f'  Correct rate: {round(unit_correct_rate,3)}')
            print(f'  95% CI: ({round(lower_95,3)} ~ {round(higher_95,3)})')
            # print(f'  99% CI: ({round(lower_99,3)} ~ {round(higher_99,3)})')

        # 计算所有单元的准确率均值和方差
        assert len(unit_correct_rates) > 0, f'{model_name} has no unit correct answers'
        mean_accuracy = np.mean(unit_correct_rates)
        std_accuracy = np.std(unit_correct_rates, ddof=0)  # 总体标准差
        var_accuracy = np.var(unit_correct_rates, ddof=0)  # 总体方差

        print(f'Model: {model_name}')
        print(f'Total units: {len(unit_correct_rates)}')
        print(f'Mean accuracy across all units: {round(mean_accuracy, 4)}')
        print(f'Variance of accuracy: {round(var_accuracy, 4)}')
        print(f'Std dev of accuracy: {round(std_accuracy, 4)}')

        print('\n' + '=' * 60)

    print('OK')