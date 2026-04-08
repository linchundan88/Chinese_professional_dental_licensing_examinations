from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

def get_llm_client(model_name, timeout=30):

    if model_name in ['qwen-max',  'qwen3-max', 'deepseek-v3.1', 'deepseek-v3.2', 'qwen-plus', 'qwen3.5-plus']:
        client = OpenAI(
            base_url=os.getenv("SERVICE_URL_ali"),
            api_key=os.getenv("API_KEY_ALI"),
            timeout=timeout
        )
    elif model_name in [ 'gpt-3.5-turbo-instruct', 'gpt-3.5-turbo', 'gpt-4', 'gpt-5', 'gpt-5.4-pro',
        'gpt-5.1', 'gpt-5.1-chat-latest', 'gpt-5.2-chat-latest', 'gpt-5.1', 'gpt-5-chat-latest', 'gpt-5.4', 'gpt-5.4-pro',
                       'gemini-2.5-pro', 'gemini-2.5-pro-thinking', 'gemini-3-pro-preview', 'gemini-3.1-pro-preview', #
                         'doubao-seed-1.6', 'doubao-seed-1-8-251228','doubao-seed-2-0-pro-260215',
                         'deepseek-v3', 'deepseek-v3.2',
                        'claude-sonnet-4-5']:
        client = OpenAI(
            base_url=os.getenv("SERVICE_URL_ZZZ"),
            api_key=os.getenv("API_KEY_ZZZ"),
            timeout=timeout
        )
    # elif model_name in ['doubao-seed-2-0-pro-260215']:
    #     client = OpenAI(
    #         base_url=os.getenv("SERVICE_URL_HUOSHAN"),
    #         api_key=os.getenv("API_KEY_HUOSHAN"),
    #     )

    # elif model_name in ['gpt-5.4-pro']:
    #     client = OpenAI(
    #         base_url=os.getenv("SERVICE_URL_JIEKOU"),
    #         api_key=os.getenv("API_KEY_JIEKOU"),
    #     )

    elif model_name in ['deepseek-v3',  'deepseek-r1', 'deepseek-chat']:
        client = OpenAI(
            base_url=os.getenv("SERVICE_URL_DEEPSEEK"),
            api_key=os.getenv("API_KEY_DEEPSEEK"),
            timeout=timeout
        )
    else:  # locally deployed models using ollama
        client = OpenAI(
            base_url=os.getenv("SERVICE_URL_LOCAL"),
            api_key='ollama',  # required, but unused
            timeout=timeout
        )

    return client