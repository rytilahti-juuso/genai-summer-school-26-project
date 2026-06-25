import json
import re
from llm_api_connection import LLM_connection
from secret_key import local_key2
from mockUpData import MOCK_ARXIV_RECORDS


llama4 = "llama4:latest"
llm = LLM_connection(llama4, local_key2)
with open('clusterlabelingrecords.json') as json_data:
    clusters = json.loads(json_data.read())
    

label_tag = r"<LABEL>\s*([^<]+)\s*(?:</LABEL>)?"
cluster_tag = r"<CLUSTER>\s*([^<]+)\s*(?:</CLUSTER>)?"

result_list = []


def get_tags(output: str, tag):
    pattern = re.compile(tag)
    match = pattern.search(output)
    if match:
        return match.group(1).strip()
    return None

def label_clusters(llm, clusters):
	for cluster in clusters:
		prompt_text = {
			"task": "Analyze the cluster and create a category label for the articles in the cluster.",
  			"description": " First, explain your reasoning and then present the label. Don't do anything else. I want only one label for the cluster and I want it to be short and concise. I want the label inside <LABEL> tags. I want you to also ouput the cluster number inside <CLUSTER> tags.",
  			"input": cluster,
  			}
		result = llm.invoke(json.dumps(prompt_text))
		label = get_tags(result, label_tag)
		cluster_number = get_tags(result, cluster_tag)
		result_kv = {"cluster": cluster_number, "label": label}
		result_list.append(result_kv)
		print(cluster_number + " " + label)
	return result_list

label_clusters(llm, clusters)

