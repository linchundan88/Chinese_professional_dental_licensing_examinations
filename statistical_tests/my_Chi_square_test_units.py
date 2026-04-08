'''
Treat the result of each question as a binary classification outcome
performance comparison of the same model for two different units
Chi square test
'''
import numpy as np
from pathlib import Path
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--examination_type', default='Professional_Licensing_Examination')  # Professional_Licensing_Examination  Assistant_Professional_Licensing_Examination
parser.add_argument('--instruction_no', type=int, default=0)
parser.add_argument('--temperature', type=int, default=0)
parser.add_argument('--thinking_suffix', default='')  # /think  /no_think
parser.set_defaults(unknown_as_error=False)
args = parser.parse_args()
from libs.my_helper_exam import parse_result
from scipy.stats import chi2_contingency


if __name__ == "__main__":
    list_model_name = ['qwen-max', 'qwen3-max',  # 'qwen3.5-max',
                       'gpt-3.5-turbo', 'gpt-4', 'gpt-5', 'gpt-5.4',
                       'doubao-seed-1.6', 'doubao-seed-1-8-251228', 'doubao-seed-2-0-pro-260215',
                       'gemini-2.5-pro', 'gemini-3-pro-preview', 'gemini-3.1-pro-preview']

    # list_model_name = ['doubao-seed-1.6']

    # list_unit1 = [1, 2, 3]
    list_unit1 = [2]
    list_unit2 = [4]

    alpha = 0.05

    for model_name in list_model_name:
        print(f'model name: {model_name}')

        filename_pkl1 = f'{args.examination_type}_{model_name.replace(":", "_")}_instruction_no{args.instruction_no}_{args.thinking_suffix}.pkl'
        with open(Path(__file__).resolve().parents[1] / 'results' / f'temperature_{args.temperature}' / filename_pkl1, 'rb') as f1:
            df1, list_prediction_answer = pickle.load(f1)
            list_correct_answers = df1['答案'].tolist()
            list_units = df1['单元编号'].tolist()

        list_prediction_answer = [parse_result(answer) for answer in list_prediction_answer]

        list_correct_answers1 = []
        for answer1 in list_correct_answers:
            answer1 = answer1.strip()
            answer1 = answer1[0:1]
            list_correct_answers1.append(answer1)

        for unit_num1 in list_unit1:
            for unit_num2 in list_unit2:
                if unit_num1 == unit_num2:
                    continue

                print(f'do Chi square test for {model_name} on unit:{unit_num1} and unit:{unit_num2}' )

                # region do Chi square test

                # 筛选单元编号为1 和 2 的数据
                unit1_indices = [i for i, unit in enumerate(list_units) if unit == unit_num1]
                unit2_indices = [i for i, unit in enumerate(list_units) if unit == unit_num2]

                # 提取单元 1 和单元 2 的预测答案和正确答案
                prediction_unit1 = [list_prediction_answer[i] for i in unit1_indices]
                correct_unit1 = [list_correct_answers1[i] for i in unit1_indices]

                prediction_unit2 = [list_prediction_answer[i] for i in unit2_indices]
                correct_unit2 = [list_correct_answers1[i] for i in unit2_indices]

                # 计算每个单元的混淆矩阵（正确/错误）
                correct_unit1 = [1 if pred == corr else 0 for pred, corr in zip(prediction_unit1, correct_unit1)]
                incorrect_unit1 = [1 - c for c in correct_unit1]

                correct_unit2 = [1 if pred == corr else 0 for pred, corr in zip(prediction_unit2, correct_unit2)]
                incorrect_unit2 = [1 - c for c in correct_unit2]

                # 构建列联表
                #        正确  错误
                # 单元 1   a     b
                # 单元 2   c     d
                a = sum(correct_unit1)  # 单元 1 正确数
                b = sum(incorrect_unit1)  # 单元 1 错误数
                c = sum(correct_unit2)  # 单元 2 正确数
                d = sum(incorrect_unit2)  # 单元 2 错误数

                contingency_table = np.array([[a, b], [c, d]])

                # print(f'\n列联表:')
                # print(f'        正确  错误')
                # print(f'单元{unit_num1}:   {a:4d}  {b:4d}')
                # print(f'单元{unit_num2}:   {c:4d}  {d:4d}')

                # 计算准确率
                acc_unit1 = a / (a + b) * 100 if (a + b) > 0 else 0
                acc_unit2 = c / (c + d) * 100 if (c + d) > 0 else 0
                # print(f'\n单元{unit_num1}准确率：{acc_unit1:.2f}%')
                # print(f'单元{unit_num2}准确率：{acc_unit2:.2f}%')

                # 进行卡方检验
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)

                print(f'\n卡方检验结果:')
                # print(f'卡方值 (χ²): {chi2:.4f}')
                # print(f'自由度 (df): {dof}')
                print(f'p 值：{p_value:.6f}')
                # print(f'期望频数:\n{expected}')

                # if p_value < alpha:
                #     print(f'\n结论：在α={alpha} 水平下，两个单元的答题表现存在显著差异 (p={p_value:.6f})')
                # else:
                #     print(f'\n结论：在α={alpha} 水平下，两个单元的答题表现无显著差异 (p={p_value:.6f})')

                # endregion


    print('OK')