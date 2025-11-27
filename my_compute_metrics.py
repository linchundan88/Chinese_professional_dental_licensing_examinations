'''

'''

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--filename_pkl', default='Assistant_Professional_Licensing_Examination_deepseek-v3.1_.pkl')
args = parser.parse_args()
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from pathlib import Path
from libs.my_helper_ststistics import my_bootstrap
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import random

def parse_result(prediction):
    if '</think>' in prediction:
        prediction = re.sub(r"<think>.*?</think>\n?", "", prediction, flags=re.DOTALL)

    for replace_str in ['*', ':', '：', ' ', '\n']:
        prediction = prediction.replace(replace_str, '')

    if prediction.strip() in ['A', 'B', 'C', 'D', 'E']:
        return prediction.strip()
    if prediction.strip() in ['a', 'b', 'c', 'd', 'e']:
        return prediction.strip().upper()
    if prediction.strip() == '1':
        return 'A'
    if prediction.strip() == '2':
        return 'B'
    if prediction.strip() == '3':
        return 'C'
    if prediction.strip() == '4':
        return 'D'
    if prediction.strip() == '5':
        return 'E'

    list_patterns = [r'正确答案选项是([ABCDE])', r'选项([ABCDE])', r'答案选项([ABCDE])', r'正确答案是选项([ABCDE])',
                    r'正确答案([ABCDE])', r'正确答案是([ABCDE])', r'正确答案编号([ABCDE])', r'正确答案为选项([ABCDE])', r'答案选项([ABCDE])',
                    r'答案([ABCDE])',  r'答案为选项([ABCDE])',  r'正确答案的编号是([ABCDE])',  r'正确答案的选项是([ABCDE])', r'最正确的答案是选项([ABCDE])']

    prediction_str = prediction[0: 20]
    for pattern in list_patterns:
        match = re.search(pattern, prediction_str)
        if match:
            predicted_answer = match.group(1)
            return predicted_answer

    if len(prediction) > 40:
        prediction_str = prediction[-20:]
        for pattern in list_patterns:
            match = re.search(pattern, prediction_str)
            if match:
                predicted_answer = match.group(1)
                return predicted_answer

    # print(prediction)

    error_msg = f"Error: data parsing error. {prediction}"  # sometimes return no correct answer. 无正确答案
    return error_msg



if __name__ == '__main__':

    filename_pkl = args.filename_pkl
    print(args.filename_pkl)

    random.seed(800)
    np.random.seed(800)

    with open(filename_pkl, 'rb') as f:
        df, list_prediction_answer = pickle.load(f)

    list_correct_answers = df['答案'].tolist()
    list_predictions = []
    number_errors = 0

    for correct_answer, prediction_answer in zip(list_correct_answers, list_prediction_answer):
        correct_answer = correct_answer.strip()
        prediction_answer = parse_result(prediction_answer)

        if prediction_answer in ['A', 'B', 'C', 'D', 'E']:
            if correct_answer.strip() == prediction_answer.strip():
                list_predictions.append(1)
            else:
                # print(f'correct_answer:{correct_answer}, prediction_answer:{prediction_answer}')
                list_predictions.append(0)
        else:
            number_errors += 1
            list_predictions.append(0)  # Human can parse answer from the prediction answer, but in all cases the answers are incorrect.


    correct_rate = np.mean(list_predictions)
    lower_correct_rate_95, higher_correct_rate_95 = my_bootstrap(list_predictions, np.mean, c=0.95)
    lower_correct_rate_99, higher_correct_rate_99 = my_bootstrap(list_predictions, np.mean, c=0.99)

    print(f'number of errors: {number_errors}')
    print(f'correct_rate:{correct_rate:.4f}')
    print(f'(lower_correct_rate_95:{lower_correct_rate_95:.4f}~higher_correct_rate_95:{higher_correct_rate_95:.4f})')
    print(f'(lower_correct_rate_99:{lower_correct_rate_99:.4f}~higher_correct_rate_99:{higher_correct_rate_99:.4f})')


'''
Professional_Licensing_Examination_qwen-plus_
number of errors: 0
correct_rate:0.8512
(lower_correct_rate_95:0.8367~higher_correct_rate_95:0.8646)
(lower_correct_rate_99:0.8317~higher_correct_rate_99:0.8700)


Assistant_Professional_Licensing_Examination_qwen-plus_
number of errors: 1  # 无正确答案
correct_rate:0.8752 
(lower_correct_rate_95:0.8591~higher_correct_rate_95:0.8919)
(lower_correct_rate_99:0.8538~higher_correct_rate_99:0.8999)


Professional_Licensing_Examination_deepseek-v3.1_
number of errors: 34
correct_rate:0.7271
(lower_correct_rate_95:0.7092~higher_correct_rate_95:0.7450)
(lower_correct_rate_99:0.7029~higher_correct_rate_99:0.7521)


Assistant_Professional_Licensing_Examination_deepseek-v3.1_
number of errors: 18
correct_rate:0.7523
(lower_correct_rate_95:0.7316~higher_correct_rate_95:0.7757)
(lower_correct_rate_99:0.7190~higher_correct_rate_99:0.7804)

'''