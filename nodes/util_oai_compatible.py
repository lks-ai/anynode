import os
import requests

class OpenAICompatible:
    def __init__(self, api_key, endpoint="https://api.openai.com/v1/chat/completions", model="gpt-4o"):
        self.api_key = api_key
        self.api_url = endpoint
        self.model = model

    def get_response(self, prompt, any=None, temperature=0.1, max_tokens=2048):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": "You are an AI assistant."},
                         {"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

# Example Usage:
# api_key = os.getenv("OPENAI_API_KEY")
# openai_compatible = OpenAICompatible(api_key=api_key)
# response = openai_compatible.get_response(prompt="Please describe this image in 10 to 20 words.")
# print(response)
