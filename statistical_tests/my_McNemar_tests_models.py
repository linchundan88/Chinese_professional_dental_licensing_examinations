'''
Treat the result of each question as a binary classification outcome
performance comparison of two different models on the same unit (or all units)
McNemar test for paired data
'''
import numpy as np
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar
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


if __name__ == "__main__":
    model_name1 = 'doubao-seed-2-0-pro-260215'
    model_name2 = 'doubao-seed-1-8-251228'

    unit_num = None

    # list_model_name1 = ['qwen-max', 'qwen3-max',  # 'qwen3.5-max',
    #                    'gpt-3.5-turbo', 'gpt-4', 'gpt-5', 'gpt-5.4',
    #                    'doubao-seed-1.6', 'doubao-seed-1-8-251228', 'doubao-seed-2-0-pro-260215',
    #                    'gemini-2.5-pro-thinking', 'gemini-3-pro-preview', 'gemini-3.1-pro-preview']
    # list_model_name2 = ['qwen-max', 'qwen3-max',  # 'qwen3.5-max',
    #                    'gpt-3.5-turbo', 'gpt-4', 'gpt-5', 'gpt-5.4',
    #                    'doubao-seed-1.6', 'doubao-seed-1-8-251228', 'doubao-seed-2-0-pro-260215',
    #                    'gemini-2.5-pro-thinking', 'gemini-3-pro-preview', 'gemini-3.1-pro-preview']

    list_model_name1 = ['doubao-seed-2-0-pro-260215']
    list_model_name2 = ['qwen-max', 'qwen3-max',  # 'qwen3.5-max',
                       'gpt-3.5-turbo', 'gpt-4', 'gpt-5', 'gpt-5.4',
                       'doubao-seed-1.6', 'doubao-seed-1-8-251228', 'doubao-seed-2-0-pro-260215',
                       'gemini-2.5-pro-thinking', 'gemini-3-pro-preview', 'gemini-3.1-pro-preview']

    for model1 in list_model_name1:
        for model2 in list_model_name2:
            if model1 == model2:
                continue

            print(f'{model_name1}  vs  {model_name2}')

            filename_pkl1 = f'{args.examination_type}_{model_name1.replace(":", "_")}_instruction_no{args.instruction_no}_{args.thinking_suffix}.pkl'
            with open(Path(__file__).resolve().parents[1] / 'results' / f'temperature_{args.temperature}' / filename_pkl1, 'rb') as f1:
                df1, list_prediction_answer1 = pickle.load(f1)
                list_correct_answers = df1['答案'].tolist()
                # list_exams = df1['试卷编号'].tolist()
                list_units = df1['单元编号'].tolist()
            filename_pkl2 = f'{args.examination_type}_{model_name2.replace(":", "_")}_instruction_no{args.instruction_no}_{args.thinking_suffix}.pkl'
            with open(Path(__file__).resolve().parents[0] / 'results' / f'temperature_{args.temperature}' / filename_pkl2, 'rb') as f2:
                df2, list_prediction_answer2 = pickle.load(f2)

            list_prediction_answer1 = [parse_result(answer) for answer in list_prediction_answer1]
            list_prediction_answer2 = [parse_result(answer) for answer in list_prediction_answer2]

            list_correct_answers1 = []
            for answer1 in list_correct_answers:
                answer1 = answer1.strip()
                answer1 = answer1[0:1]
                list_correct_answers1.append(answer1)

            for unit_num in [1, 2, 3, 4, None]:
                pass

            num_both_correct , num_both_incorrect = 0, 0
            num_answer1_correct_answer2_incorrect, num_answer1_incorrect_answer2_correct = 0, 0

            for correct_answer, unit1, answer1, answer2 in zip(list_correct_answers1, list_units, list_prediction_answer1, list_prediction_answer2):
                if unit_num is not None:
                    if unit1 != unit_num:
                        continue

                answer1 = parse_result(answer1)
                answer2 = parse_result(answer2)

                if answer1 == correct_answer and answer2 == correct_answer:
                    # print('both correct')
                    num_both_correct += 1
                elif answer1 == correct_answer and answer2 != correct_answer:
                    # print('answer1 correct, answer2 incorrect')
                    num_answer1_correct_answer2_incorrect += 1
                elif answer1 != correct_answer and answer2 == correct_answer:
                    # print('answer1 incorrect, answer2 correct')
                    num_answer1_incorrect_answer2_correct += 1
                elif answer1 != correct_answer and answer2 != correct_answer:
                    # print('both incorrect')
                    # print(answer1, answer2)
                    num_both_incorrect += 1

            #region do McNemar's test


            # 1. 准备 2x2 列联表
            # 格式：
            # [[都为正/对的频数,  仅A为正/对的频数],
            #  [仅B为正/对的频数, 都为负/错的频数]]

            data_table = np.array([[num_both_correct, num_answer1_correct_answer2_incorrect],
                                   [ num_answer1_incorrect_answer2_correct, num_both_incorrect]])
            # 2. 执行 McNemar 检验
            # 参数 exact=False 表示使用卡方分布近似（适用于大样本，b+c > 25）
            # 参数 correction=True 表示使用 Edwards 连续性校正（推荐开启）
            result = mcnemar(data_table, exact=False, correction=True)

            print(f"统计量 (Chi-Squared): {result.statistic:.4f}")
            print(f"P 值 (p-value): {result.pvalue:.4f}")

            # 解释结果
            alpha = 0.05
            if result.pvalue < alpha:
                print("拒绝原假设：两者存在显著差异")
            else:
                print("不能拒绝原假设：两者无显著差异")

            # endregion

    print('OK')