from dotenv import load_dotenv
load_dotenv("configs/.env")
import os
import re
import json
import ast

from typing import Annotated, Optional, Union, Dict, Any
from openai._types import NOT_GIVEN
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from functools import wraps

def create_response_format(schema: dict) -> dict:
    """
    根据给定的 schema 字典快速生成 OpenAI API 的 response_format 参数。

    参数 schema 示例:
    {
        "字段名": {"type": "数据类型", "description": "字段描述"},
        ...
    }
    """
    properties = {}
    for key, val in schema.items():
        prop = {"type": val["type"], "description": val["description"]}
        # 添加数组类型的items定义
        if val["type"] == "array" and "items" in val:
            prop["items"] = val["items"]
        properties[key] = prop

    json_schema = {
        "name": "response_jsons",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": list(schema.keys()),
            "additionalProperties": False
        }
    }

    return {
        "type": "json_schema", 
        "json_schema": json_schema
    }

def retry_with_exponential_backoff(
    max_retries: int = 2,
    base_delay: float = 2,
    max_delay: float = 2
):
    """
    指数退避重试装饰器
    """
    def decorator(func):
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=base_delay, max=max_delay),
            reraise=False
        )
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
            
        @wraps(func)
        def safe_wrapper(*args, **kwargs):
            try:
                return wrapper(*args, **kwargs)
            except RetryError as e:
                last_exception = e.last_attempt.exception()
                return f"在重试 {max_retries} 次后仍然失败。错误: {str(last_exception)}"
            except Exception as e:
                return f"发生未预期的错误: {str(e)}"
                
        return safe_wrapper
    return decorator

class AzureGPT4Chat:
    def __init__(
            self,
            system_prompt="You are a helpfule assistant.",
            model_name=None
    ):
        import traceback
        print("AzureGPT4Chat 被调用")
        print("调用栈如下：")
        traceback.print_stack()
        from openai import AzureOpenAI, OpenAI
        if os.getenv("DASHSCOPE_API_KEY"):
            self.client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            self.deployment_name = model_name or "qwen-plus"
            self.is_azure = False
        elif os.getenv("SILICONFLOW_API_KEY"):
            self.client = OpenAI(
                api_key=os.getenv("SILICONFLOW_API_KEY"),
                base_url="https://api.siliconflow.cn/v1"
            )
            self.deployment_name = model_name or "deepseek-ai/DeepSeek-R1"
            self.is_azure = False
        elif os.getenv("AZURE_OPENAI_API_KEY"):
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-12-01-preview",
                azure_endpoint='https://18449-m91k0g5r-swedencentral.openai.azure.com/'
            )
            self.deployment_name = "gpt-4o-mini"
            self.is_azure = True
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.deployment_name = model_name or "gpt-4o-mini"
            self.is_azure = False

        self.system_prompt = system_prompt

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    @retry_with_exponential_backoff()
    def chat(self, question, system_prompt=None):
        messages = [
            {"role": "system", "content": self.system_prompt if system_prompt is None else system_prompt},
            {"role": "user", "content": question}
        ]
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages
        )
        return response.choices[0].message.content
    
    @retry_with_exponential_backoff()
    def chat_with_message(self, message, model_name=None):
        if model_name is None:
            model_name = self.deployment_name
        response = self.client.chat.completions.create(
            model=model_name,
            messages=message
        )
        return response.choices[0].message.content

    @retry_with_exponential_backoff()
    def chat_with_message_format(
        self, 
        question=None,
        system_prompt=None, 
        message_list=None,
        response_format=None
    ):
        """
        使用指定的输出格式进行对话
        
        Args:
            question (str): 用户问题
            response_format (dict): 响应格式,例如 {"type": "json_object"} 或 {"type": "text"}
            system_prompt (str, optional): 可选的系统提示
        """        
        if message_list is None:
            messages = [
                {"role": "system", "content": self.system_prompt if system_prompt is None else system_prompt},
                {"role": "user", "content": question}
            ]
        else:
            messages = message_list
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            response_format=response_format if response_format else NOT_GIVEN
        )
        print(response)
        return response.choices[0].message.content

    def parse_llm_response(self, response_text: str) -> Dict:
        """
        Parse LLM response text into dictionary.
        """
        # Remove any markdown code block indicators
        response_text = re.sub(r"```(?:json|python)?\s*", "", response_text)
        response_text = response_text.strip("`")

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(response_text)
            except (SyntaxError, ValueError):
                result = {}
                pattern = r'["\']?(\w+)["\']?\s*:\s*([^,}\n]+)'
                matches = re.findall(pattern, response_text)
                for key, value in matches:
                    try:
                        result[key] = ast.literal_eval(value)
                    except (SyntaxError, ValueError):
                        result[key] = value.strip("\"'")
                return result

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("/Users/chloe/Documents/Academic/AI/Project/基于Colpali的多模态检索标准框架/multimodal-RAG/DeepRAG_Multimodal/configs/.env")
    system_m = """
    You are a language model tasked with enhancing queries for a multimodal document retrieval system. For every input query, you need to perform query expansion for each of the following three task types: ["Understanding", "Reasoning", "Locating"]. For each task type, please do the following:

    1. Expand the Query: Generate an expanded version of the input query that is tailored to the specific task type.
    2. Judgment Probability: Provide a probability (a decimal between 0 and 1) indicating how likely it is that the input query belongs to that task type. Ensure that the sum of probabilities for all three task types is exactly 1.
    3. Expansion Reasoning: Write a one-sentence explanation that justifies the expanded query. The explanation should follow the format:
    "This expansion [details of the expansion], indicating the task type of [task type]."

    Return your response in valid JSON format. The JSON structure should have keys corresponding to each task type ("Understanding", "Reasoning", "Locating"), and each task type should include the following keys:
    - `"expanded_query"`
    - `"probability"`
    - `"explanation"`

    Output Example:

    ```json
    {
    "question": "What role does Dr. Michaela C. Fried play in the document?",
    "Understanding": {
        "expanded_query": "Provide a detailed explanation of what role Dr. Michaela C. Fried plays in the document and how her position influences the content.",
        "probability": 0.4,
        "explanation": "This expansion emphasizes understanding the significance of Dr. Fried's role and its impact on the document, indicating the task type of Understanding."
    },
    "Reasoning": {
        "expanded_query": "Analyze and deduce the implications of Dr. Michaela C. Fried's involvement in the document to interpret her influence on the overall narrative.",
        "probability": 0.35,
        "explanation": "This expansion explores logical implications and influences to infer the importance of Dr. Fried's role in shaping the document's message, indicating the task type of Reasoning."
    },
    "Locating": {
        "expanded_query": "Search for specific sections in the document where Dr. Michaela C. Fried's role is described or referenced.",
        "probability": 0.25,
        "explanation": "This expansion is aimed at locating precise parts of the document that detail the role of Dr. Fried, indicating the task type of Locating."
    }
    }
    ```

    Make sure your output is strictly in JSON format and that each task type includes all three required keys with appropriate values.
    """
    agent = AzureGPT4Chat(system_prompt=system_m)
    jsonl_path = '/home/liuguanming/multimodal-RAG/LongDocURL/LongDocURL_public_with_subtask_category.jsonl'
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    answer = []
    for item in data:
        try:
            multimodal_queries = agent.expand_multimodal_query(item['question'])
            print(f"多模态查询扩展结果: {multimodal_queries}")
            result = agent.chat(item['question'])
            answer.append(agent.parse_llm_response(result))
        except:
            continue
    answer = json.dumps(answer, ensure_ascii=False, indent=4)
    with open('query_expansion_task.jsonl', 'w', encoding='utf-8') as f:
        f.write(answer)