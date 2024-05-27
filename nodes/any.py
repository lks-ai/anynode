"""
@author: newsbubbles
@title: AnyNode v0.1
@nickname: AnyNode
@description: This single node uses an LLM to generate a functionality based on your request. You can make the node do anything.
"""
# inspired by: rgthree AnySwitch, shouts and thanks to MachineLearners Discord

# TODO: finish inputs section which shows python types of inputs for the final prompt
# - Store the json in some sort of hidden file which stores in the saved workflow
# - How do I use just the "internal variables" since it seems scope on AnyNode class params are global (not good for storing script)

import os, json, random, string, sys, math, datetime, collections, itertools, functools, urllib, shutil, re, torch, time
import numpy
import numpy as np
import torch.nn.functional as F
import collections.abc
import traceback
import os
import openai
from openai import OpenAI
# from .context_utils import is_context_empty, _create_context_data
# from .constants import get_category, get_name
from .utils import any_type, is_none, variable_info
from .util_gemini import GoogleGemini
from .util_oai_compatible import OpenAICompatible

# The template for the system message sent to ChatCompletions
SYSTEM_TEMPLATE = """
# Coding a Python Function
You are an expert python coder who specializes in writing custom nodes for ComfyUI.

## Imports you may use
It is not required to use any of these libraries, but if you do use any import in your code it must be on this list:
[[IMPORTS]]

## Input Data
Here is some important information about the input data:
- input_data: [[INPUT]]
[[CODEBLOCK]]
## Coding Instructions
- Your job is to code the user's requested node given the input and desired output type.
- Respond with only the code in one function named generated_function that takes one argument named 'input_data' and wraps any other functions you might use.
- Write only the code contents of the function itself.
- Do include needed imports in your code before the function.
- If an input is a Tensor and the output is a Tensor, it should be the same shape unless otherwise specified by the user.
- Your resulting code should be as compute efficient as possible.
- If there is a code block above, be sure to only write the new version of the code without any commentary or banter.

## Write the Code
```python
"""

class AnyNode:
  """Ask it to make up any node for you. """

  NAME = "AnyNode"
  CATEGORY = "utils"

  ALLOWED_IMPORTS = {"os", "re", "json", "random", "string", "sys", "math", "datetime", "collections", "itertools", "functools", "urllib", "shutil", "numpy", "openai", "traceback", "torch", "time"}

  def __init__(self):
      self.script = None
      self.last_prompt = None
      self.imports = []
      self.last_error = None
  
  @classmethod
  def INPUT_TYPES(self):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {
        "prompt": ("STRING", {
          "multiline": True,
          "default": "Take the input and multiply by 5",
        }),
      },
      "optional": {
        "any": (any_type,),
      },
    }

  @classmethod
  def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
      return time.time()

  RETURN_TYPES = (any_type,)
  RETURN_NAMES = ('any',)
  FUNCTION = "go"
  
  def render_template(self, template:str, any=None, seed=None):
      """Render the system template with current state"""
      varinfo = variable_info(any)
      print(f"LE: {self.last_error}")
      instruction = "" if not self.last_error else f"There was an error with the current code.\n\n### Traceback\n{self.last_error}\n\n### Erroneous Code"
      print(f"Input 0 -> {varinfo}")
      r = template \
          .replace('[[IMPORTS]]', ", " \
          .join(list(self.ALLOWED_IMPORTS))).replace('[[INPUT]]', varinfo) \
          .replace("[[CODEBLOCK]]", "" if not self.script else f"\n## Current Code:\n{instruction}\n```python\n{self.script}\n```\n")
      # This is the case where we call from error mitigation
      return r

  def get_response(self, system_message:str, prompt:str) -> str:
      client = OpenAI(
          # This is the default and can be omitted
          api_key=os.environ.get("OPENAI_API_KEY"),
      )
      response = client.chat.completions.create(
          model="gpt-4o",  # Use the model of your choice, e.g., gpt-4 or gpt-3.5-turbo
          messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
          ]
        )
        # Extract the response text
      return response.choices[0].message.content.strip().replace('```python', '').replace('```', '')

  def get_llm_response(self, prompt:str, any=None) -> str:
      """Calls OpenAI and returns response"""
      try:
          final_template = self.render_template(SYSTEM_TEMPLATE, any=any)
          print("\n", final_template)
          return self.get_response(final_template, prompt)
      except Exception as e:
          return f"An error occurred: {e}"

  def extract_imports(self, generated_code):
      """
      Extracts import statements from the generated code and stores them in self.imports.
      Returns the code without the import statements.
      """
      import_pattern = re.compile(r'^\s*(import .+|from .+ import .+)', re.MULTILINE)
      imports = import_pattern.findall(generated_code)
      cleaned_code = import_pattern.sub('', generated_code).strip()
      
      # Store the imports in the instance variable
      self.imports = [imp.strip() for imp in imports]
      print(f"Imports in code: {self.imports}")
      return cleaned_code

  def safe_exec(self, code_string, globals_dict=None, locals_dict=None):
      """Execute """
      if globals_dict is None:
          globals_dict = {}
      if locals_dict is None:
          locals_dict = {}
          
      try:
          exec(code_string, globals_dict, locals_dict)
      except Exception as e:
          print("An error occurred:")
          traceback.print_exc()
          raise e

  def go(self, prompt:str, any=None):
      """Takes the prompt and inputs, then """
      result = None
      if not is_none(any):
          print(f"Last Error: {self.last_error}")
          if self.script is None or self.last_prompt != prompt or self.last_error is not None:
              print("Generating Node function...")
              # Generate the function code using OpenAI
              r = self.get_llm_response(prompt, any=any)
              
              # Store the script for future use
              self.script = self.extract_imports(r)
              print(f"Stored script:\n{self.script}")
              self.last_prompt = prompt

          # Define a dictionary to store globals and locals, updating it with imported libs from script and built in functions
          globals_dict = {"__builtins__": __builtins__}
          globals_dict.update({imp.split()[1]: globals()[imp.split()[1]] for imp in self.imports if imp.startswith('import')})
          globals_dict.update({imp.split()[1]: globals()[imp.split()[3]] for imp in self.imports if imp.startswith('from')})
          globals_dict.update({"np": np})
          locals_dict = {}

          # Execute the stored script to define the function
          try:
              self.safe_exec(self.script, globals_dict, locals_dict)
          except Exception as e:
              # store the error for next run
              self.last_error = traceback.format_exc()
              if 'not defined' in self.last_error:
                  # case where a library is missing
                  raise e
              else:
                  raise e

          # Assuming the generated code defines a function named 'generated_function'
          function_name = "generated_function"
          if function_name in locals_dict:
              try:
                  # Call the generated function and get the result
                  result = locals_dict[function_name](any)
                  print(f"Function result: {result}")
              except Exception as e:
                  print(f"Error calling the generated function: {e}")
                  traceback.print_exc()
                  self.last_error = traceback.format_exc()
                  raise e
          else:
              print(f"Function '{function_name}' not found in generated code.")
      
      self.last_error = None              
      return (result,)
  
class AnyNodeGemini(AnyNode):
    def __init__(self, api_key):
        super().__init__()
        self.llm = GoogleGemini(api_key)

    def get_response(self, prompt, any=None):
        return self.llm.get_response(prompt, any=any)

class AnyNodeOpenAICompatible(AnyNode):
    def __init__(self, api_key):
        super().__init__()
        self.llm = OpenAICompatible(api_key)

    def get_response(self, prompt, any=None):
        return self.llm.get_response(prompt, any=any)

    
NODE_CLASS_MAPPINGS = {
    "AnyNode": AnyNode,
    "AnyNodeGemini": AnyNodeGemini,
    "AnyNodeLocal": AnyNodeOpenAICompatible,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyNode": "Any Node 🍄",
    "AnyNodeGemini": "Any Node 🍄 (Gemini)",
    "AnyNodeLocal": "Any Node 🍄 (Local LLM)",
}

if __name__ == "__main__":
    node = AnyNode()
    example_prompt = "Generate a random number using the input as seed"
    example_any = 5
    result = node.go(prompt=example_prompt, any=example_any)
    print("Result:", result)
