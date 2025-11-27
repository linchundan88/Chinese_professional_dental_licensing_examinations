import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--filename_pkl', default='Professional_Licensing_Examination_qwen3_32b_instruction_no1_.pkl')
args = parser.parse_args()
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from libs.my_helper_exam import parse_result
import pickle
import random

if __name__ == '__main__':

    filename_pkl = args.filename_pkl
    print('Compute performance metrics of every exam', args.filename_pkl)

    random.seed(800)
    np.random.seed(800)

    list_options = ['A', 'B', 'C', 'D', 'E']

    with open(Path(__file__).resolve().parents[0] / 'results' / filename_pkl, 'rb') as f:
        df, list_prediction_answer = pickle.load(f)

    list_exam_no = df['试卷编号'].tolist()
    list_correct_answers = df['答案'].tolist()

    exam_count = len(set(list_exam_no))
    list_exam_predictions = [[] for _ in range(exam_count)]
    list_exam_correct_rate = [0 for _ in range(exam_count)]
    list_exam_num_parsing_errors = [0 for _ in range(exam_count)]

    for exam_no, correct_answer, prediction_answer in zip(list_exam_no, list_correct_answers, list_prediction_answer):
        exam_no -= 1  # exam_no starts from 1 in the xlsx file.

        correct_answer = correct_answer.strip()
        prediction_answer = parse_result(prediction_answer)

        if prediction_answer in list_options:
            if correct_answer.strip() == prediction_answer.strip():
                list_exam_predictions[exam_no].append(1)
            else:
                list_exam_predictions[exam_no].append(0)
        else:
            list_exam_num_parsing_errors[exam_no] += 1  # both parsing errors and unknown instances are considered as errors.

    for exam_no in range(exam_count):
        list_exam_correct_rate[exam_no] = np.mean(list_exam_predictions[exam_no])

    list_exam_correct_rate = [f'{correct_rate:.4f}' for correct_rate in list_exam_correct_rate]
    print('correct rate of each exam:', list_exam_correct_rate)
    print('num of parsing errors of each exam:', list_exam_num_parsing_errors)

'''

'''