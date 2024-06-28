import anthropic

class AnthropicClaude:
    def __init__(self, model:str="claude-3-5-sonnet-20240620"):
        self.model = model

    def get_response(self, system:str, prompt:str, any=None, temperature:float=0.0, max_tokens:int=2048):
        try:
            response = anthropic.Anthropic().messages.create(
                model=self.model,
                max_tokens=1024,
                system=system, # <-- system prompt
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            print('GOT ANTHROPIC')
            print(response)
            r = response.content[0].text
            print(response.content)
            print(r)
            return r
        except Exception as e:
            raise e

