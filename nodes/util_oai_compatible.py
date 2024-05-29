import os
import requests

class OpenAICompatible:
    def __init__(self, api_key=None, endpoint="https://api.openai.com", model="gpt-4o"):
        self.api_key = api_key
        self.set_api_server(endpoint)
        self.model = model

    def set_api_server(self, endpoint:str):
        self.api_url = endpoint + "/v1/chat/completions"

    def get_response(self, system:str, prompt:str, any=None, temperature:float=0.0, max_tokens:int=2048):
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

if __name__ == "__main__":
    # Example Usage:
    #api_key = os.getenv("OPENAI_API_KEY")
    openai_compatible = OpenAICompatible(endpoint="http://localhost:11434", api_key='ollama', model="mistral")
    response = openai_compatible.get_response("You are an AI assistant.", "Where is the eiffel tower?")
    print(response)
