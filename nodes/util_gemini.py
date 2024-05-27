import os
import requests

class GoogleGemini:
    def __init__(self, api_key, model:str="gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://gemini.googleapis.com/v1/models/{self.model}:generateText"

    def get_response(self, prompt, any=None, temperature=0.7, max_tokens=100):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json().get('choices', [{}])[0].get('text', '').strip()
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

# Example Usage:
# api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
# gemini = GoogleGemini(api_key=api_key)
# response = gemini.get_response(prompt="Please describe this image in 10 to 20 words.")
# print(response)
