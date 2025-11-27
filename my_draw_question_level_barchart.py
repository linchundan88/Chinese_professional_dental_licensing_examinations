import matplotlib.pyplot as plt
import seaborn as sns
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

    if args.exam_type == 'dental_licensing_examination':
        correct_rates = [0.8654, 0.8512, 0.7271, 0.7408, 0.6512]

        # 95% confidence intervals
        ci_95_lower = [0.8521,  0.8367, 0.7092, 0.7242, 0.6338]
        ci_95_upper = [0.8783, 0.8646, 0.7450, 0.7562, 0.6708]

    if args.exam_type == 'assistant_level_dental_licensing_examination':
        correct_rates = [0.8818, 0.8752, 0.7523, 0.7630, 0.6722]

        # 95% confidence intervals
        ci_95_lower = [0.8652,  0.8591, 0.7316, 0.7430, 0.6509]
        ci_95_upper = [0.8965, 0.8919, 0.7757, 0.7844, 0.6976]

    # Calculate error bars for 95% confidence intervals
    error_95 = [correct_rates[i] - ci_95_lower[i] for i in range(len(models))], \
               [ci_95_upper[i] - correct_rates[i] for i in range(len(models))]

    # Set up the plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Create bar plot with 95% confidence intervals
    bars = plt.bar(models, correct_rates, yerr=error_95, capsize=5,
                   color=['skyblue', 'lightgreen', 'salmon', 'orange'],
                   alpha=0.7, edgecolor='black', linewidth=0.5)

    # Customize the plot
    plt.ylabel('Correct Rate', fontsize=12)
    plt.title(' Question-Level Accuracy Rate Comparison with 95% Confidence Intervals', fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')

    # Add legend
    plt.legend([bars[0]], ['95% Confidence Interval'], loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{args.exam_type}_question_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print('OK.')