'''

'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--filename_pkl', default='Assistant_Professional_Licensing_Examination_qwen3_32b_instruction_no1_.pkl')
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

if __name__ == '__main__':

    filename_pkl = args.filename_pkl
    print('Compute performance metrics based on all questions', args.filename_pkl)

    random.seed(800)
    np.random.seed(800)

    list_options = ['A', 'B', 'C', 'D', 'E']

    with open(Path(__file__).resolve().parents[0] / 'results' / filename_pkl, 'rb') as f:
        df, list_prediction_answer = pickle.load(f)

    num_records = len(df)

    list_correct_answers = df['答案'].tolist()
    list_predictions = []
    number_unknown, number_parsing_errors = 0, 0
    number_correct, number_errors = 0, 0

    for correct_answer, prediction_answer in zip(list_correct_answers, list_prediction_answer):
        correct_answer = correct_answer.strip()
        prediction_answer = parse_result(prediction_answer)

        if prediction_answer in ['我不知道']:
            number_unknown += 1
        elif prediction_answer in list_options:
            if correct_answer.strip() == prediction_answer.strip():
                number_correct += 1
                list_predictions.append(1)
            else:
                # print(f'correct_answer:{correct_answer}, prediction_answer:{prediction_answer}')
                number_errors += 1
                list_predictions.append(0)
        else:
            # parsing errors are considered as error answers.
            number_parsing_errors += 1
            list_predictions.append(0)

    print(f'number of parsing errors: {number_parsing_errors}, percent of parsing errors: {number_parsing_errors / num_records:.4%}')
    print(f'number of unknown: {number_unknown}, percent of error unknown: {number_unknown / num_records:.4%}')
    print(f'number of correct answers: {number_correct}, percent of correct answers: {number_correct / num_records:.4%}')
    print(f'number of error answers: {number_errors}, percent of error answers: {number_errors / num_records:.4%}')

    correct_rate = np.mean(list_predictions)
    lower_correct_rate_95, higher_correct_rate_95 = my_bootstrap(list_predictions, np.mean, c=0.95)
    lower_correct_rate_99, higher_correct_rate_99 = my_bootstrap(list_predictions, np.mean, c=0.99)

    print('parsing errors are considered as errors.')
    print(f'correct_rate:{correct_rate:.4f}')
    print(f'(lower_correct_rate_95:{lower_correct_rate_95:.4f}~higher_correct_rate_95:{higher_correct_rate_95:.4f})')
    print(f'(lower_correct_rate_99:{lower_correct_rate_99:.4f}~higher_correct_rate_99:{higher_correct_rate_99:.4f})')

    # unknown instances are considered as errors.
    if number_unknown > 0:
        list_unknown = [0 for _ in range(number_unknown)]
        list_predictions.extend(list_unknown)

        correct_rate_1 = np.mean(list_predictions)  # both parsing errors and unknown instances are considered as errors.
        lower_correct_rate_95_1, higher_correct_rate_95_1 = my_bootstrap(list_predictions, np.mean, c=0.95)
        lower_correct_rate_99_1, higher_correct_rate_99_1 = my_bootstrap(list_predictions, np.mean, c=0.99)

        print('both parsing errors and unknown instances are considered as errors.')
        print(f'correct_rate_1:{correct_rate_1:.4f}')
        print(f'(lower_correct_rate_95_1:{lower_correct_rate_95_1:.4f}~higher_correct_rate_95:{higher_correct_rate_95_1:.4f})')
        print(f'(lower_correct_rate_99_1:{lower_correct_rate_99_1:.4f}~higher_correct_rate_99:{higher_correct_rate_99_1:.4f})')


'''
Professional_Licensing_Examination_qwen-plus_instruction_no0_.pkl
number of errors: 0
correct_rate:0.8512
(lower_correct_rate_95:0.8367~higher_correct_rate_95:0.8646)
(lower_correct_rate_99:0.8317~higher_correct_rate_99:0.8700)


Assistant_Professional_Licensing_Examination_qwen-plus_instruction_no0_.pkl
number of errors: 1  # 无正确答案
correct_rate:0.8752 
(lower_correct_rate_95:0.8591~higher_correct_rate_95:0.8919)
(lower_correct_rate_99:0.8538~higher_correct_rate_99:0.8999)


Professional_Licensing_Examination_deepseek-v3.1_instruction_no0_.pkl
number of errors: 34
correct_rate:0.7271
(lower_correct_rate_95:0.7092~higher_correct_rate_95:0.7450)
(lower_correct_rate_99:0.7029~higher_correct_rate_99:0.7521)


Assistant_Professional_Licensing_Examination_deepseek-v3.1_instruction_no0_.pkl
number of errors: 18
correct_rate:0.7523
(lower_correct_rate_95:0.7316~higher_correct_rate_95:0.7757)
(lower_correct_rate_99:0.7190~higher_correct_rate_99:0.7804)


Professional_Licensing_Examination_gpt-oss_120b_instruction_no0_.pkl
number of errors: 1  #ACE
correct_rate:0.6512
(lower_correct_rate_95:0.6338~higher_correct_rate_95:0.6708)
(lower_correct_rate_99:0.6238~higher_correct_rate_99:0.6746)

Assistant_Professional_Licensing_Examination_gpt-oss_120b_.pkl
number of errors: 1  #3,4
correct_rate:0.6722
(lower_correct_rate_95:0.6509~higher_correct_rate_95:0.6976)
(lower_correct_rate_99:0.6395~higher_correct_rate_99:0.7056)


Professional_Licensing_Examination_qwen3_32b_instruction_no0_.pkl
number of errors: 24
correct_rate:0.7408
(lower_correct_rate_95:0.7242~higher_correct_rate_95:0.7562)
(lower_correct_rate_99:0.7146~higher_correct_rate_99:0.7633)

Professional_Licensing_Examination_qwen3-max_instruction_no0_.pkl
number of errors: 1
correct_rate:0.8654
(lower_correct_rate_95:0.8521~higher_correct_rate_95:0.8783)
(lower_correct_rate_99:0.8467~higher_correct_rate_99:0.8838)

Professional_Licensing_Examination_qwen3-max_instruction_no1_.pkl
number of errors: 0
number of unknown: 1
correct_rate:0.9412
(lower_correct_rate_95:0.8824~higher_correct_rate_95:0.9882)
(lower_correct_rate_99:0.8706~higher_correct_rate_99:1.0000)

Professional_Licensing_Examination_qwen-plus_instruction_no1_.pkl
number of errors: 0
number of unknown: 1
correct_rate:0.9257
(lower_correct_rate_95:0.8959~higher_correct_rate_95:0.9517)
(lower_correct_rate_99:0.8810~higher_correct_rate_99:0.9665)

Professional_Licensing_Examination_deepseek-v3.1_instruction_no1_.pkl
number of errors: 1
number of unknown: 1
correct_rate:0.7368
(lower_correct_rate_95:0.5263~higher_correct_rate_95:0.9474)
(lower_correct_rate_99:0.4737~higher_correct_rate_99:0.9474)

'''