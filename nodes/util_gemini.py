import os
import requests
import google.generativeai as genai

class GoogleGemini:
    def __init__(self, model:str="gemini-1.5-flash"):
        self.model = model

    def get_response(self, system:str, prompt:str, any=None, temperature:float=0.0, max_tokens:int=2048):
        model = genai.GenerativeModel(self.model)
        chat = model.start_chat(history=[
            {
                "role": "user",
                "parts": [{"text": system}]
            },
        ])
        response = chat.send_message(prompt, generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=max_tokens,
                temperature=temperature)
        )
        return response.text
        # response = requests.post(self.api_url, headers=headers, params={"key": self.api_key}, json=payload)
        # if response.status_code == 200:
        #     return response.json().get('content', [{}])[0].get('text', '').strip()
        # else:
        #     raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

# Example Usage:
# api_key = os.getenv("GOOGLE_API_KEY")
# gemini = GoogleGemini(api_key=api_key)
# response = gemini.get_response(system="You are an assistant.", prompt="Please describe this image in 10 to 20 words.")
# print(response)
