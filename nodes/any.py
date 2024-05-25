"""
AnyNode v0.1
author: newsbubbles
inspired by: rgthree AnySwitch
shouts and thanks: MachineLearners Discord
"""

# TODO: finish inputs section which shows python types of inputs for the final prompt

import os, json, random, sys, math, datetime, collections, itertools, functools, urllib, shutil, re, numpy
import traceback
import os
import openai
from openai import OpenAI
from .context_utils import is_context_empty, _create_context_data
from .constants import get_category, get_name
from .utils import any_type

def is_none(value):
    """Checks if a value is none. Pulled out in case we want to expand what 'None' means."""
    if value is not None:
        if isinstance(value, dict) and 'model' in value and 'clip' in value:
            return is_context_empty(value)
    return value is None

SYSTEM_TEMPLATE = """
# Coding a Python Function
You are an expert python coder who specializes in writing custom nodes for ComfyUI.

## Coding Instructions
- Your job is to code the user's requested node given the input and desired output type.
- Code only the contents of the function itself.
- Respond with only the code in a function named generated_function that takes one argument named 'input_data'.

## Write the Code
```python
"""

class AnyNode:
  """Ask it to make up any node for you. """

  NAME = "AnyNode"
  CATEGORY = "utils"
  
  script = None
  last_prompt = None
  imports = []
  
  ALLOWED_IMPORTS = {"os", "re", "json", "random", "sys", "math", "datetime", "collections", "itertools", "functools", "urllib", "shutil", "numpy", "openai", "traceback"}

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
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

  RETURN_TYPES = (any_type,)
  RETURN_NAMES = ('any',)
  FUNCTION = "go"

  def get_openai_response(self, prompt):
      try:
          client = OpenAI(
              # This is the default and can be omitted
              api_key=os.environ.get("OPENAI_API_KEY"),
              api_base=os.environ.get("ANYNODE_ENDPOINT"),
          )
          response = client.chat.completions.create(
              model="gpt-4o",  # Use the model of your choice, e.g., gpt-4 or gpt-3.5-turbo
              messages=[
                  {"role": "system", "content": sys_prompt},
                  {"role": "user", "content": prompt}
              ]
          )
          # Extract the response text
          return response.choices[0].message.content.strip().replace('```python', '').replace('```', '')
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
      
      return cleaned_code

  def safe_exec(self, code_string, globals_dict=None, locals_dict=None):
      if globals_dict is None:
          globals_dict = {}
      if locals_dict is None:
          locals_dict = {}
          
      try:
          exec(code_string, globals_dict, locals_dict)
      except Exception as e:
          print("An error occurred:")
          traceback.print_exc()

  def go(self, prompt, any=None):
      """Anything"""
      result = None
      if not is_none(any):
          if self.script is None or self.last_prompt != prompt:
              # Generate the function code using OpenAI
              r = self.get_openai_response(prompt)
              print(f"Generated code:\n{r}")
              
              # Store the script for future use
              self.script = self.extract_imports(r)
              print(f"Stored script:\n{self.script}")
              self.last_prompt = prompt

          # Define a dictionary to store globals and locals
          #globals_dict = {"__builtins__": {}}
          globals_dict = {
              "__builtins__": {},
              "os": os,
              "json": json,
              "random": random,
              "sys": sys,
              "math": math,
              "datetime": datetime,
              "collections": collections,
              "itertools": itertools,
              "functools": functools,
              "urllib": urllib,
              "shutil": shutil,
              "re": re,
              "numpy": numpy,
              "openai": openai,
              "traceback": traceback,
          }

          locals_dict = {}

          # Execute the stored script to define the function
          self.safe_exec(self.script, globals_dict, locals_dict)

          # Assuming the generated code defines a function named 'generated_function'
          function_name = "generated_function"
          if function_name in locals_dict:
              try:
                  # Call the generated function and get the result
                  result = locals_dict[function_name](any)
                  #result = locals_dict[function_name]()
                  print(f"Function result: {result}")
              except Exception as e:
                  print(f"Error calling the generated function: {e}")
                  traceback.print_exc()
          else:
              print(f"Function '{function_name}' not found in generated code.")
              
      return (result,)
    
NODE_CLASS_MAPPINGS = {
    "AnyNode": AnyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyNode" : "Any Node 🍄",
}

if __name__ == "__main__":
    node = AnyNode()
    example_prompt = "Generate a random number using the input as seed"
    example_any = 5
    result = node.go(prompt=example_prompt, any=example_any)
    print("Result:", result)
