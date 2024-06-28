import os
import requests
import google.generativeai as genai

class GoogleGemini:
    def __init__(self, model:str="gemini-1.5-flash"):
        self.model = model

    def get_response(self, system:str, prompt:str, any=None, temperature:float=0.0, max_tokens:int=2048):
        try:
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
        except Exception as e:
            raise e
