import json

import requests
from loguru import logger


def generate(system: str, user: str, **kwargs) -> str:
    """
    """
    # Configure URL
    url = "http://10.10.178.25:12239/aigateway/google/chat/completions"
    # Configure request headers
    headers = {
        "content-type": "application/json;charset=utf-8",
    }
    # Configure request data
    payload = {
        "body": {
            "model": "gemini-2.5-pro",
            "maxOutputTokens": kwargs.get("max_tokens", 16384),
            "temperature": kwargs.get("temperature", 0.6),
            "stream": False,
            "contents": [
                {"role": "system", "parts": [{"text": system}]},
                {"role": "user", "parts": [{"text": user}]},
            ],
            "tools": kwargs.get("tools", [{"functionDeclarations": []}]),
        },
        "PKey": "MDlGQTM0RUZFOUYxREY5Njk4MzQyQzcwNDQ1MkIxMDY=",
        "source": "Wind.AI.Insight",
    }
    # Send POST request
    response = requests.post(url=url, json=payload, headers=headers)
    # Handle request failure
    if response.status_code != 200:
        raise Exception("Model inference failed!" f"HTTP status code: {response.status_code}", f"Response data: {response.text}")
    # Parse response data
    answer_content = ""
    response_data = response.json()
    try:
        content = response_data["body"]["candidates"][0]["content"]["parts"][0]["text"]
        answer_content = content
        function_call = response_data["body"]["candidates"][0]["content"]["parts"][0]["functionCall"]
    except Exception as e:
        if "Alice audit service call failed!" in response_data.get("message", ""):
            raise PermissionError("Alice audit service call failed!", response_data) from e
        logger.error(f"Model inference error! Error reason: {e}, Response data: {response_data}")
        raise Exception("Model inference error!", response_data)
    return answer_content


def stream_generate(system: str, user: str, **kwargs) -> str:
    """
    """
    # Configure URL
    url = "http://10.10.178.25:12239/aigateway/google/chat/completions"
    # Configure request headers
    headers = {
        "content-type": "application/json;charset=utf-8",
    }
    # Configure request data
    payload = {
        "body": {
            "model": "gemini-2.5-pro",
            "max_tokens": kwargs.get("max_tokens", 16384),
            "temperature": kwargs.get("temperature", 0.6),
            "stream": True,
            "contents": [
                {"role": "system", "parts": [{"text": system}]},
                {"role": "user", "parts": [{"text": user}]},
            ],
        },
        "PKey": "MDlGQTM0RUZFOUYxREY5Njk4MzQyQzcwNDQ1MkIxMDY=",
        "source": "Wind.AI.Insight",
    }
    # Send POST request
    response = requests.post(url=url, json=payload, headers=headers, stream=True)
    # Handle request failure
    if response.status_code != 200:
        raise Exception("Model inference failed!" f"HTTP status code: {response.status_code}", f"Response data: {response.text}")
    # Parse response data
    answer_content = ""
    for line in response.iter_lines(decode_unicode=True):
        line: str
        if not line:
            continue
        elif line.startswith("data: "):
            line = line.lstrip("data: ")
        elif "Alice audit service call failed!" in line:
            raise PermissionError("Alice audit service call failed!", line)
        else:
            raise Exception("Model inference error!", line)
        try:
            data_blk = json.loads(line)
            content = data_blk["candidates"][0]["content"]["parts"][0]["text"]
            answer_content += content
            print(content, end="", flush=True)
        except Exception as e:
            logger.error(f"Model inference error! Error reason: {e}, Response data: {line}")
    if answer_content:
        print(flush=True)
    return answer_content