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

    prediction_str = prediction

    list_patterns = [r'正确答案选项是([ABCDE])', r'选项([ABCDE])', r'答案选项([ABCDE])', r'正确答案是选项([ABCDE])',
                     r'正确答案是([ABCDE])', r'选择([ABCDE])', r'正确答案为([ABCDE])',
                    r'正确答案([ABCDE])', r'正确答案是([ABCDE])', r'正确答案编号([ABCDE])', r'正确答案为选项([ABCDE])', r'答案选项([ABCDE])',
                    r'答案([ABCDE])',  r'答案为选项([ABCDE])',  r'正确答案的编号是([ABCDE])',  r'正确答案的选项是([ABCDE])', r'最正确的答案是选项([ABCDE])']
    list_patterns_unknown = [r'我不知道']

    for pattern in list_patterns:
        match = re.search(pattern, prediction_str)
        if match:
            predicted_answer = match.group(1)
            return predicted_answer

    list_patterns = [r'[ABCDE]',]
    for pattern in list_patterns:
        match = re.search(pattern, prediction_str)
        if match:
            predicted_answer = match.group(0)
            return predicted_answer

    for pattern in list_patterns_unknown:
        match = re.search(pattern, prediction_str)
        if match:
            predicted_answer = match.group(0)
            return predicted_answer

    print(prediction_str)
    error_msg = f"Error: data parsing error. {prediction_str}"  # sometimes return no correct answer. 无正确答案
    return error_msg


def process_llm_prediction(record1, chat_client, model_name, str_instruction, input_text_prefix, thinking_suffix='',
                           temperature=1, max_completion_tokens=500, list_options=None, index=-1):
    if list_options is None:
        list_options = ['A', 'B', 'C', 'D', 'E']

    input_text = input_text_prefix

    input_text += record1['题干'].replace(' ', '') + ':' + '\n'
    for option in list_options:
        input_text += f'选项{option}：' + str(record1[f"选项{option}"]).replace(' ', '') + '\n'

    if thinking_suffix != '':
        input_text += thinking_suffix

    try:
        completion = chat_client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'system', 'content': str_instruction},
                      {'role': 'user', 'content': input_text}],
            # extra_body={"enable_thinking": False}   # useless
            # https://platform.openai.com/docs/api-reference/completions/create#chat-create-stop
            # A Temperature of 0, a Top K of 1, or a Top P of 0 is the same as replacing softmax with the argmax formula.
            temperature=temperature,  # it is a scaling factor applied to the log-probs (or logits) prior to the softmax application.
            # A high value for Temperature squeezes all of our options closer to each other, so they have a closer probability, or a small value for Temperature stretches them apart, making the probabilities of each option further apart.
            # Top_p sampling is an alternative to temperature sampling. Instead of considering all possible tokens, GPT-3 considers only a subset of tokens (the nucleus) whose cumulative probability mass adds up to a certain threshold (top_p).
            # top_p=1,  # default value of 1
            max_completion_tokens=max_completion_tokens  # Limits total generated tokens (output + reasoning)
        )
    except Exception as e:
        error_msg = f"System Error: {e}"
        return error_msg

    if completion.choices is None:
        return 'Error: completion.choices is None'


    prediction = completion.choices[0].message.content

    if  index == -1:
        return prediction
    else:
        return index, prediction

list_instructions = ["您是一个资深的口腔医生，请根据问题描述从给出的五个选项中选择一个正确的答案，请直接回答该正确答案的编号 A,B,C,D,或者E，也就是只需要回答一个答案编号的字母。",
                     "您是一个资深的口腔医生，请根据问题描述从给出的五个选项中选择一个正确的答案，如果您比较确定那个选项是正确的请直接回答该正确答案的编号A,B,C,D,或者E，也就是只需要回答一个答案编号的字母，否则回答我不知道。",
                     "您是一个资深的口腔医生，请根据问题描述从给出的五个选项中选择一个正确的答案。 如果您有90%以上的把握度能够正确回答该问题则请直接回答该正确答案的编号A,B,C,D,或者E，也就是只需要回答一个答案编号的字母，否则回答我不知道。如果回答正确得1分，回答我不知道得0分，回答错误得负1分"
                     ]
