'''
Treat the exam scores as continuous values.
performance comparison of two different models on the same unit (or all units)
paired t test
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
from scipy import stats
from libs.my_helper_exam import parse_result


if __name__ == '__main__':

    model_name1 = 'doubao-seed-2-0-pro-260215'
    model_name2 = 'doubao-seed-1-8-251228'

    unit_num = None

    # list_model_name1 = ['qwen-max', 'qwen3-max',  # 'qwen3.5-max',
    #                    'gpt-3.5-turbo', 'gpt-4', 'gpt-5', 'gpt-5.4',
    #                    'doubao-seed-1.6', 'doubao-seed-1-8-251228', 'doubao-seed-2-0-pro-260215',
    #                    'gemini-2.5-pro-thinking', 'gemini-3-pro-preview', 'gemini-3.1-pro-preview']

    list_model_name1 = ['doubao-seed-2-0-pro-260215']
    list_model_name2 = ['qwen-max', 'qwen3-max',  # 'qwen3.5-max',
                       'gpt-3.5-turbo', 'gpt-4', 'gpt-5', 'gpt-5.4',
                       'doubao-seed-1.6', 'doubao-seed-1-8-251228',
                       'gemini-2.5-pro-thinking', 'gemini-3-pro-preview', 'gemini-3.1-pro-preview']


    for model1 in list_model_name1:
        filename_pkl1 = f'{args.examination_type}_{model_name1.replace(":", "_")}_instruction_no{args.instruction_no}_{args.thinking_suffix}.pkl'
        with open(Path(__file__).resolve().parents[1] / 'results' / f'temperature_{args.temperature}' / filename_pkl1, 'rb') as f1:
            df1, list_prediction_answer1 = pickle.load(f1)
            list_correct_answers = df1['答案'].tolist()
            # list_exams = df1['试卷编号'].tolist()
            list_units = df1['单元编号'].tolist()

        list_correct_answers1 = []
        for answer1 in list_correct_answers:
            answer1 = answer1.strip()
            answer1 = answer1[0:1]
            list_correct_answers1.append(answer1)

        for model2 in list_model_name2:
            if model1 == model2:
                continue

            filename_pkl2 = f'{args.examination_type}_{model_name2.replace(":", "_")}_instruction_no{args.instruction_no}_{args.thinking_suffix}.pkl'
            with open(Path(__file__).resolve().parents[0] / 'results' / filename_pkl2, 'rb') as f2:
                df2, list_prediction_answer2 = pickle.load(f2)

            list_prediction_answer1 = [parse_result(answer) for answer in list_prediction_answer1]
            list_prediction_answer2 = [parse_result(answer) for answer in list_prediction_answer2]

            # 计算每个模型在每个问题上的得分（正确为 1，错误为 0）
            scores_model1 = []
            scores_model2 = []

            for i in range(len(list_correct_answers1)):
                # 模型 1 的得分
                if list_prediction_answer1[i] == list_correct_answers1[i]:
                    scores_model1.append(1)
                else:
                    scores_model1.append(0)

                # 模型 2 的得分
                if list_prediction_answer2[i] == list_correct_answers1[i]:
                    scores_model2.append(1)
                else:
                    scores_model2.append(0)

            # 进行配对 T 检验（单边检验：检验模型 1 是否大于模型 2）
            t_statistic, p_value = stats.ttest_rel(scores_model1, scores_model2, alternative='greater')

            # 计算均值和标准差
            mean1 = np.mean(scores_model1)
            std1 = np.std(scores_model1, ddof=1)
            mean2 = np.mean(scores_model2)
            std2 = np.std(scores_model2, ddof=1)

            print(f'模型 1: {model1}')
            print(f'模型 2: {model2}')
            # print(f'样本量：{len(scores_model1)}')
            # print(f'模型 1 - 准确率：{mean1:.4f}, 标准差：{std1:.4f}')
            # print(f'模型 2 - 准确率：{mean2:.4f}, 标准差：{std2:.4f}')
            # print(f'T 统计量：{t_statistic:.4f}')
            print(f'P 值：{p_value:.4f}')

            # if p_value < 0.05:
            #     print('结果：在α=0.05 水平上差异显著')
            #     if mean1 > mean2:
            #         print(f'结论：{model1}的表现显著优于{model2}')
            #     else:
            #         print(f'结论：{model2}的表现显著优于{model1}')
            # else:
            #     print('结果：在α=0.05 水平上差异不显著')
            #     print(f'结论：{model1}和{model2}的表现无显著差异')
            print('-' * 80)
            print('\n')



    print('OK')