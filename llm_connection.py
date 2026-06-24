from llm_api_connection import LLM_connection
from secret_key import local_key2
from mockUpData import MOCK_ARXIV_RECORDS

llama4 = "llama4:latest"
llm = LLM_connection(llama4, local_key2)



prompt_text = f"Explain what categories of science are in the article summaries in {MOCK_ARXIV_RECORDS}"

result = llm.invoke(prompt_text)
print(result)