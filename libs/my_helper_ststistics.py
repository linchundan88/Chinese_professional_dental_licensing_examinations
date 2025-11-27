import numpy as np
from scipy.stats import bootstrap
from sklearn.metrics import mean_squared_error, confusion_matrix


def my_bootstrap(data, func, sampling_times=100, c=0.95):  # default sampling_times changed from 500 to 1000.
    if type(data) == tuple:
        data1, data2 = data

        data1 = np.array(data1)
        data2 = np.array(data2)

        samples_num = len(data1)
        list_sample_result = []
        for i in range(sampling_times):
            index_arr = np.random.randint(0, samples_num, samples_num)  # Sampling with replacement
            sample_result = func(data1[index_arr], data2[index_arr])
            list_sample_result.append(sample_result)

    if type(data) == list:
        data1 = np.array(data)
        samples_num = len(data1)
        list_sample_result = []
        for i in range(sampling_times):
            index_arr = np.random.randint(0, samples_num, samples_num)
            sample_result = func(data1[index_arr])
            list_sample_result.append(sample_result)

    a = 1 - c
    k1 = int(sampling_times * a / 2)
    k2 = int(sampling_times * (1 - a / 2))
    auc_sample_arr_sorted = sorted(list_sample_result)
    lower = auc_sample_arr_sorted[k1]
    higher = auc_sample_arr_sorted[k2]

    return lower, higher




def compute_metrics_with_ci(data, metric_func, n_resamples=1000):
    # data (y_true, y_pred)
    res = bootstrap(data, metric_func, n_resamples=n_resamples, confidence_level=0.95, random_state=42, method='percentile')
    return metric_func(data, res.confidence_interval.low, res.confidence_interval.high)


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)
