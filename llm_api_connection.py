import requests
import json




class LLM_connection:
    def __init__(self, model, api_key):
        self.model = model
        self.api_key = api_key
        self.url = 'https://llm.tt.utu.fi/api/generate'

    def invoke(self, prompt_text):
        
        prompt = prompt_text
        url = self.url

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type':'application/json'
        }
        payload = {
            'model': self.model,
            'prompt': prompt_text,
            'stream': False,
            'options':  {
                "num_ctx": 65536,
                "num_pred": 10000
            }
        }

        response = requests.post(self.url, headers=headers, json=payload)
        return response.json()["response"]
