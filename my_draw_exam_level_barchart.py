import matplotlib.pyplot as plt
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exam_type', default='dental_licensing_examination')  # dental_licensing_examination assistant_level_dental_licensing_examination
args = parser.parse_args()


if __name__ == '__main__':

    # Model performance data
    models = [
        'Qwen3-Max',
        'Qwen-Plus',
        'DeepSeek-V3.1',
        'Qwen3_32B',
        'GPT-OSS_120B'
    ]

    print('exam type:', args.exam_type)
    if args.exam_type == 'dental_licensing_examination':
        group_label = [f'Exam{i}' for i in range(1, 5)]

        scores_qwen3_max = [0.9217, 0.7129, 0.8967, 0.9317]
        scores_qwen_plus = [0.9100, 0.7233, 0.8600, 0.9117]
        scores_deepseek = [0.7597, 0.6142, 0.7661, 0.8098]
        scores_qwen3_32b = [0.8030, 0.5990, 0.7798, 0.8105]
        score_gpt_oss_120b = [0.7028, 0.5150, 0.6767, 0.7117]

        title_str = 'Model Performance per Exam of Dental Licensing Examination '
        xlabel = 'Dental Licensing Exams'

    if args.exam_type == 'assistant_level_dental_licensing_examination':
        group_label = [f'Exam{i}' for i in range(1, 6)]

        scores_qwen3_max = [0.8967, 0.8733, 0.8833, 0.8859, 0.8729]
        scores_qwen_plus = [0.8900, 0.8600, 0.8700, 0.8926, 0.8662]
        scores_deepseek = [0.8087, 0.7416, 0.7703, 0.7543, 0.7322]
        scores_qwen3_32b = [0.7483, 0.7864, 0.7744, 0.7819, 0.7576]
        score_gpt_oss_120b = [0.6488, 0.6533, 0.7133, 0.7013, 0.6467]

        title_str = 'Model Performance per Exam of Assistant Level Dental Licensing Examination '
        xlabel = 'Assistant Level Dental Licensing Exams'

    print('qwen3_max', np.mean(scores_qwen3_max), np.std(scores_qwen3_max))
    print('qwen_plus', np.mean(scores_qwen_plus), np.std(scores_qwen_plus))
    print('deepseek', np.mean(scores_deepseek), np.std(scores_deepseek))
    print('qwen3_32b', np.mean(scores_qwen3_32b), np.std(scores_qwen3_32b))
    print('gpt_oss_120b', np.mean(score_gpt_oss_120b), np.std(score_gpt_oss_120b))


    # Organize data for plotting
    all_scores = [scores_qwen3_max, scores_qwen_plus, scores_deepseek, scores_qwen3_32b, score_gpt_oss_120b]

    # Set up the plot
    x = np.arange(len(group_label))  # Label locations
    width = 0.15  # Width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create bars for each model
    rects1 = ax.bar(x - 2 * width, scores_qwen3_max, width, label='Qwen3-Max')
    rects2 = ax.bar(x - width, scores_qwen_plus, width, label='Qwen-Plus')
    rects3 = ax.bar(x, scores_deepseek, width, label='DeepSeek-V3.1')
    rects4 = ax.bar(x + width, scores_qwen3_32b, width, label='Qwen3_32B')
    rects5 = ax.bar(x + 2 * width, score_gpt_oss_120b, width, label='GPT-OSS_120B')

    # Add labels, title, and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Scores')
    ax.set_title(title_str)
    ax.set_xticks(x)
    ax.set_xticklabels(group_label)
    ax.legend()


    # Add value labels on bars
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.3f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)


    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)

    fig.tight_layout()

    plt.savefig(f'{args.exam_type}_exam_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # print('OK.')