'''
two-proportion z-test
用于比较两个单元(unit)的答题正确率是否有显著差异
'''
import numpy as np
from pathlib import Path
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--examination_type', default='Professional_Licensing_Examination')  # Professional_Licensing_Examination  Assistant_Professional_Licensing_Examination
parser.add_argument('--instruction_no', type=int, default=0)
parser.add_argument('--temperature', type=int, default=1)
parser.add_argument('--thinking_suffix', default='')  # /think  /no_think
parser.set_defaults(unknown_as_error=False)
args = parser.parse_args()
from statsmodels.stats.proportion import proportions_ztest
from libs.my_helper_exam import parse_result


if __name__ == '__main__':


    list_unit1 = [1, 2, 3]
    list_unit2 = [4]

    list_model_name1 = ['qwen-max', 'qwen3-max',  # 'qwen3.5-max',
                       'gpt-3.5-turbo', 'gpt-4', 'gpt-5', 'gpt-5.4',
                       'doubao-seed-1.6', 'doubao-seed-1-8-251228','doubao-seed-2-0-pro-260215',
                       'gemini-2.5-pro-thinking', 'gemini-3-pro-preview', 'gemini-3.1-pro-preview']

    list_correct_answers_all = []
    list_prediction_answers_all = []
    list_units_all = []

    for model1 in list_model_name1:
        print(f'model:{model1}')
        filename_pkl1 = f'{args.examination_type}_{model1.replace(":", "_")}_instruction_no{args.instruction_no}_{args.thinking_suffix}.pkl'
        with open(Path(__file__).resolve().parents[1] / 'results' / f'temperature_{args.temperature}' / filename_pkl1, 'rb') as f1:
            df1, list_prediction_answer1 = pickle.load(f1)
            list_correct_answers = df1['答案'].tolist()
            # list_exams = df1['试卷编号'].tolist()
            list_units = df1['单元编号'].tolist()

        list_units_all.extend(list_units)

        list_correct_answers1 = []
        for answer1 in list_correct_answers:
            answer1 = answer1.strip()
            answer1 = answer1[0:1]
            list_correct_answers1.append(answer1)
        list_correct_answers_all.extend(list_correct_answers1)

        list_prediction_answer = [parse_result(answer) for answer in list_prediction_answer1]
        list_prediction_answers_all.extend(list_prediction_answer)

    for unit_num1 in list_unit1:
        for unit_num2 in list_unit2:
            if unit_num1 == unit_num2:
                continue

            print(f'unit:{unit_num1} compared with unit:{unit_num2}')

            # 提取 unit_num1和 unit_num2 的预测结果
            list_predictions_unit1 = []
            list_predictions_unit2 = []

            for i, unit in enumerate(list_units_all):
                if unit == unit_num1:
                    pred1 = parse_result(list_prediction_answers_all[i])
                    if pred1 in ['A', 'B', 'C', 'D', 'E']:
                        list_predictions_unit1.append(1 if pred1 == list_correct_answers_all[i] else 0)
                elif unit == unit_num2:
                    pred2 = parse_result(list_prediction_answers_all[i])
                    if pred2 in ['A', 'B', 'C', 'D', 'E']:
                        list_predictions_unit2.append(1 if pred2 == list_correct_answers_all[i] else 0)


            # 计算每个单元的正确数和总数
            n1 = np.sum(np.array(list_units_all) == unit_num1)  # unit1 的样本数
            n2 = np.sum(np.array(list_units_all) == unit_num2)  # unit2 的样本数
            
            if n1 == 0 or n2 == 0:
                print(f"Warning: One of the units has no valid predictions. Skipping comparison.")
                continue
            
            x1 = np.sum(np.sum(np.array(list_predictions_unit1) == 1))  # unit1 的正确数
            x2 = np.sum(np.sum(np.array(list_predictions_unit2) == 1))  # unit2 的正确数
            
            p1 = x1 / n1  # unit1 的正确率
            p2 = x2 / n2  # unit2 的正确率
            
            # 打印基本统计信息
            print(f"\nUnit {unit_num1}: n={n1}, correct={x1}, proportion={p1:.4f}")
            print(f"Unit {unit_num2}: n={n2}, correct={x2}, proportion={p2:.4f}")
            print(f"Difference in proportions (Unit {unit_num1} - Unit {unit_num2}): {p1 - p2:.4f}")

            # 执行双比例z检验 (two-proportion z-test)
            # 使用 stats.proportions_ztest 进行双侧检验
            # 然后手动计算单侧检验的p值
            count = np.array([x1, x2])  # 成功次数
            nobs = np.array([n1, n2])   # 样本大小
            
            # 执行双侧z检验

            z_statistic, p_value_two_sided = proportions_ztest(count, nobs, alternative='two-sided')
            
            # 根据备择假设转换为单侧检验的p值
            # alternative='greater' 表示检验 p1 > p2
            if p1 > p2:
                p_value_one_sided = p_value_two_sided / 2
            else:
                p_value_one_sided = 1 - p_value_two_sided / 2
            
            print(f"Z-statistic: {z_statistic:.4f}")
            print(f"P-value (one-tailed, greater): {p_value_one_sided:.4f}")

            # 判断显著性 (单边检验)
            # if p_value < 0.05:
            #     significance = "*"
            #     if p_value < 0.01:
            #         significance = "**"
            #         if p_value < 0.001:
            #             significance = "***"
            #     print(f"Significance: {significance} (Unit {unit_num1} > Unit {unit_num2}, p < 0.05)")
            # else:
            #     print(f"Not significant (Unit {unit_num1} ≤ Unit {unit_num2}, p >= 0.05)")
            #


    print('OK.')
