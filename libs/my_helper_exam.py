import re


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
    list_patterns_unknown = [r'我不知道']

    prediction_str = prediction[0: 20]
    for pattern in list_patterns:
        match = re.search(pattern, prediction_str)
        if match:
            predicted_answer = match.group(1)
            return predicted_answer

    for pattern in list_patterns_unknown:
        match = re.search(pattern, prediction_str)
        if match:
            predicted_answer = match.group(0)
            return predicted_answer

    if len(prediction) > 40:
        prediction_str = prediction[-20:]

        for pattern in list_patterns:
            match = re.search(pattern, prediction_str)
            if match:
                predicted_answer = match.group(1)
                return predicted_answer

        for pattern in list_patterns_unknown:
            match = re.search(pattern, prediction_str)
            if match:
                predicted_answer = match.group(0)
                return predicted_answer

    # print(prediction)

    error_msg = f"Error: data parsing error. {prediction}"  # sometimes return no correct answer. 无正确答案
    return error_msg
