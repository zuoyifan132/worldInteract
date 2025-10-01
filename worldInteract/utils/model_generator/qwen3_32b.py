#%%
import json

import requests
from loguru import logger


def generate(system: str, user: str, **kwargs):
    """
    Generate function with function calling support
    
    Args:
        messages: Message list
        **kwargs: Optional parameters, including tools, max_tokens, temperature, etc.
        
    Returns:
        Tuple[str, str, List[Dict]]: (thinking_content, answer_content, tool_calls)
    """
    # Configure URL
    url = "http://10.200.64.10/10-flashc-openllm/v1/chat/completions"
    # Configure request headers
    headers = {
        "content-type": "application/json;charset=utf-8"
    }
    # Configure request body
    body = {
        "model": "qwen3-32b",
        "max_tokens": kwargs.get("max_tokens", 4069),
        "temperature": kwargs.get("temperature", 0),
        "stream": False,
        "chat_template_kwargs": {
            "enable_thinking": True
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    
    # If tools are provided, add them to the request body
    tools = kwargs.get("tools", [])
    if tools:
        body["tools"] = tools
    
    # Send POST request
    response = requests.post(url=url, data=json.dumps(body), headers=headers, timeout=100000)
    # Handle request exceptions
    if response.status_code != 200:
        raise Exception("Request failed!", f"Request status code: {response.status_code}, Response data: {response.text}")
    
    # Parse response data
    thinking_content, answer_content, tool_calls = "", "", []
    response_data = response.json()
    try:
        message = response_data["choices"][0]["message"]
        # Get thinking content (adjust according to actual return fields)
        thinking_content = message.get("reasoning_content", "") or message.get("thinking_content", "")
        # Get answer content
        answer_content = message.get("content", "")
        # Get tool calls
        tool_calls = message.get("tool_calls", [])
    except Exception as exc:
        if "Alice audit service call failed!" in response_data.get("message", ""):
            raise PermissionError("Alice audit service call failed!", response_data) from exc
        logger.error("Request error!\nResponse data:\n{}\nError reason:\n{}", response_data, exc)
    
    return thinking_content, answer_content, tool_calls
    

def stream_generate(system: str, user: str, **kwargs) -> str:
    """
    Stream generation function
    """
    # Configure URL
    url = "http://10.200.64.10/10-flashc-openllm/v1/chat/completions"
    # Configure request headers
    headers = {
        "content-type": "application/json;charset=utf-8",
    }
    # Configure request body
    payload = {
        "model": "qwen3-32b",
        "max_tokens": kwargs.get("max_tokens", 16384),
        "temperature": kwargs.get("temperature", 0),
        "stream": True,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    # Send POST request
    response = requests.post(url=url, json=payload, headers=headers, stream=True)
    # Handle request exceptions
    if response.status_code != 200:
        raise Exception("Model inference failed!", f"Request status code: {response.status_code}", f"Response data: {response.text}")
    # Parse response data
    thinking_field, answer_field = "reasoning_content", "content"
    thinking_flag, answer_flag = False, False
    thinking_content, answer_content = "", ""
    for line in response.iter_lines(decode_unicode=True):
        line: str
        if not line:
            continue
        elif line.startswith("data: "):
            line = line.lstrip("data: ")
        elif "Alice audit service call failed!" in line:
            raise PermissionError("Alice audit service call failed!", line)
        else:
            pass
        try:
            if line == "[DONE]":
                break
            data_blk = json.loads(line)
            delta = data_blk.get("choices", [{}])[0].get("delta", {})
            if thinking_field in delta:
                if not thinking_flag:
                    print("<think>\n", end="", flush=True)
                    thinking_flag = True
                content = delta.get(thinking_field, "")
                thinking_content += content
                print(content, end="", flush=True)
            else:
                if thinking_flag:
                    if thinking_content:
                        print("\n</think>", flush=True)
                    thinking_flag = False
                    if not answer_flag:
                        print("<answer>\n", end="", flush=True)
                        answer_flag = True
                content = delta.get(answer_field, "")
                answer_content += content
                print(content, end="", flush=True)
        except Exception as exc:
            logger.error("Stream processing error!\nResponse data:\n{}\nError reason:\n{}", line, exc)
    if answer_content:
        print("\n</answer>", flush=True)
    return answer_content, thinking_content



if __name__ == '__main__':
    """"""
    content = generate("", "What model are you?")
    print(content)

# %%
