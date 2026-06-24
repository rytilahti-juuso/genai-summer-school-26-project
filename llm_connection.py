import json

from llm_api_connection import LLM_connection
from secret_key import local_key2
from mockUpData import MOCK_ARXIV_RECORDS


llama4 = "llama4:latest"
llm = LLM_connection(llama4, local_key2)


prompt_test = "you there?"

prompt_text = {

  "task": "analyze",
  "description": "Analyze the given input and provide structured view on what categories of science are present",
  "input": MOCK_ARXIV_RECORDS,
  "schema": {
    "fields": [
      {
        "name": "string",
        "type": "string | number | boolean | array | object",
        "description": "Analytical dimension or metric to output."
      }
    ]
  },

  "constraints": {
    "output_format": "JSON",
  }
}

result = llm.invoke(json.dumps(prompt_text))
print(result)