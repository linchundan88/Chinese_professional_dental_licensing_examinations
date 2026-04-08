import matplotlib.pyplot as plt
import seaborn as sns
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exam_type', default='dental_licensing_examination')  # dental_licensing_examination assistant_level_dental_licensing_examination
parser.add_argument('--temperature', type=int, default=1)
args = parser.parse_args()

if __name__ == '__main__':
    # Model performance data
    model_titles = ['GPT-3.5-Trubo', 'GPT-4', 'GPT-5', 'GPT-5.4',
                    'Qwen2.5-Max', 'Qwen3-Max', 'Qwen-Plus', 'Qwen3.5-Plus',  #'Qwen3.5-Max',
                    'Doubao-Seed-1.6', 'Doubao-Seed-1.8', 'Doubao-Seed-2.0-Pro',
                    'Gemini-2.5-Pro', 'Gemini-3-Pro', 'Gemini-3.1-Pro',
                    'DeepSeek-V3', 'DeepSeek-V3.2']

    if args.temperature == 1:
        correct_rates = [0.52, 0.594, 0.833, 0.814,
                         0.895, 0.917, 0.886, 0.892,
                         0.918, 0.941, 0.955,
                         0.846, 0.926, 0.932,
                         0.789, 0.811]

        # 95% confidence intervals
        ci_95_lower = [0.5,  0.574,  0.819, 0.798,
                       0.882, 0.905,  0.873, 0.879,
                       0.906,  0.931, 0.947,
                       0.832, 0.914, 0.923,
                       0.772, 0.795]
        ci_95_upper = [0.542, 0.612, 0.848, 0.829,
                       0.906, 0.928, 0.899, 0.904,
                       0.928, 0.95, 0.963,
                       0.859, 0.936, 0.942,
                       0.806, 0.827]

    elif args.temperature == 0:
        pass

    # Calculate error bars for 95% confidence intervals
    error_95 = [correct_rates[i] - ci_95_lower[i] for i in range(len(model_titles))], \
               [ci_95_upper[i] - correct_rates[i] for i in range(len(model_titles))]

    # Set up the plot
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")

    # Create bar plot with 95% confidence intervals
    # Assign colors based on model groups: 1-3 (skyblue), 4-7 (lightgreen), 8-10 (salmon), 11-13 (orange)
    bar_colors = ['skyblue'] * 4 + ['lightgreen'] * 4 + ['salmon'] * 3 + ['orange'] * 3 + ['green'] * 2
    print(bar_colors)
    bars = plt.bar(model_titles, correct_rates, yerr=error_95, capsize=8,
                   color=bar_colors,
                   alpha=0.7, edgecolor='black', linewidth=0.5)

    # Customize the plot
    plt.ylabel('Correct Rate', fontsize=14)
    plt.title(' Question level correct rates with 95% CIs', fontsize=16)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Add legend
    plt.legend([bars[0]], ['95% Confidence Interval'], loc='upper right')

    # Add passing score line at 0.6
    plt.axhline(y=0.6, color='red', linestyle='--', linewidth=2, label='Passing Score (0.6)')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{args.exam_type}_question_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print('OK.')