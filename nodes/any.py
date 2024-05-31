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
# - Security on globals
# - use re for parsing the python code instead of naively expecting the response to start with a python tag
# - fix gemini node, give it some love

import os, json, random, string, sys, math, datetime, collections, itertools, functools, urllib, shutil, re, torch, time, decimal, matplotlib, io, base64, wave, chromadb
import numpy
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from sklearn.cluster import KMeans
import collections.abc
import traceback
import os
import openai
import google.generativeai as genai
import pkgutil
import importlib
from openai import OpenAI
# from .context_utils import is_context_empty, _create_context_data
# from .constants import get_category, get_name
from .utils import any_type, is_none, variable_info, sanitize_code
from .util_gemini import GoogleGemini
from .util_oai_compatible import OpenAICompatible
from .util_functions import FunctionRegistry

# The template for the system message sent to ChatCompletions
SYSTEM_TEMPLATE = """
# Coding a Python Function
You are an expert python coder who specializes in writing custom nodes for ComfyUI.

## Available Python Modules
It is not required to use any of these libraries, but if you do use any import in your code it must be on this list:
[[IMPORTS]]

## Input Data
Here is some important information about the input data:
- input_data: [[INPUT]]
[[EXAMPLES]][[CODEBLOCK]]
## Coding Instructions
- Your job is to code the user's requested node given the input and desired output type.
- Respond with only a brief plan and the code in one function named generated_function that takes one argument named 'input_data'.
- All functions you must define should be inner functions of generated_function.
- You may briefly plan your code in plain text, but after write only the code contents of the function itself inside of a `python` code block.
- Do include needed available imports in your code before the function.
- If the request is simple enough to do without imports, like math, just do that.
- If an input is a Tensor and the output is a Tensor, it should be the same shape unless otherwise specified by the user.
- Image tensors come in the shape (batch, width, height, rgb_channels), if outputting an image, use the same shape as the input image tensor.
    - To know the tensor is an image, it will come with the last dimension as 3
    - An example image tensor for a single 512x786 image: (1, 512, 786, 3)
    - An animation is a tensor with a larger batch of images of the same shape
- Your resulting code should be as compute efficient as possible.
- Make sure to deallocate memory for anything which uses it.
- You may not use `open` or fetch files from the internet.
- If there is a code block above, be sure to only write the new version of the code without any commentary or banter.

### Example Generated function:
User: output a list of all prime numbers up to the input number
```python
import math
def generated_function(input_data):
    def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(math.sqrt(n))+1):
            if n % i == 0:
                return False
        return True
    primes = []
    num = input_data
    while num > 1:
        if is_prime(num):
            primes.append(num)
        num -= 1
    return primes
```

## Write the Code
"""

class AnyNode:
  """Ask it to make up any node for you. """

  NAME = "AnyNode"
  CATEGORY = "utils"

  ALLOWED_IMPORTS = {"os", "re", "json", "random", "string", "sys", "math", "datetime", "collections", "itertools", "functools", "numpy", "openai", "traceback", "torch", "time", "sklearn", "torchvision", "matplotlib", "io", "base64", "wave", "google.generativeai", "chromadb"}

  def __init__(self):
      self.reset()
            
  def reset(self):
      self.script = None
      self.last_prompt = None
      self.imports:list[str] = []
      self.last_error = None
      self.last_comment = None
  
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
  OUTPUT_NODE = True
  
  # TODO: Store the md5 of a prompt in function cache globally so that a duplicated node will not need to resolve
  # store function cache JSON in 'output' folder!!!!! baller.
  VERSION = "0.1.1"
  FUNCTION_REGISTRY = FunctionRegistry(schema="default", version=VERSION)
  
  def render_template(self, template:str, any=None, seed=None):
      """Render the system template with current state"""
      varinfo = variable_info(any)
      print(f"LE: {self.last_error}")
      instruction = "" if not self.last_error else f"There was an error with the last generated_function.\n\n### Debugging Instructions\n-If the error is that something is 'not defined' find a workaround using an alternative.\n- If the undefined thing is a function, most likely you didn't wrap the function inside `generated_function`.\n- Reflect on the error in your reply, be concise and accurate in analyzing the problem, then write the updated generated_function.\n- If there is a ValueError about a Dangerous construct being detected, your code has not passed the sanitizer; find an alternative.\n\n### Traceback\n{self.last_error}\n\n### Erroneous Code"
      #print(f"Input 0 -> {varinfo}")
      examples = ""
      r = template \
          .replace('[[IMPORTS]]', ", ".join(list(self.ALLOWED_IMPORTS))) \
          .replace('[[INPUT]]', varinfo) \
          .replace('[[EXAMPLES]]', examples) \
          .replace("[[CODEBLOCK]]", "" if not self.script else f"\n## Current Code\n{instruction}\n```python\n{self.script}\n```\n")
      # This is the case where we call from error mitigation
      return r

  def get_response(self, system_message:str, prompt:str, **kwargs) -> str:
      """Calls OpenAI With System Message and Prompt. Overriden in classes that extend this."""
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
      r = response.choices[0].message.content.strip()
      return r

  def get_llm_response(self, prompt:str, any=None, **kwargs) -> str:
      """Calls OpenAI and returns response"""
      try:
          print(f"INPUT {type(any)}")
          final_template = self.render_template(SYSTEM_TEMPLATE, any=any)
          print(final_template)
          r = self.get_response(final_template, prompt, **kwargs)
          #print(r)
          code_block = self.extract_code_block(r)
          #print(f"LLM COMMENTS:\n{self.last_comment}")
          return code_block
      except Exception as e:
          return f"An error occurred: {e}"

  def extract_code_block(self, response: str) -> str:
      """
      Extracts the code block from the response using regex.
      Saves everything before the code block as self.last_comment.
      Returns the code block as a string.
      """
      code_pattern = re.compile(r'(.*?)```python(.*?)```', re.DOTALL)
      match = code_pattern.search(response)
      if match:
          self.last_comment = match.group(1).strip()
          return match.group(2).strip()
      else:
          self.last_comment = response.strip('`')
          return self.last_comment

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
  
  def _prepare_globals(self, globals_dict: dict):
      """Get the globals dict prepared for safe_exec"""
      for imp in self.imports:
          parts = imp.split()
          try:
              if imp.startswith('import'):
                  # Handle 'import module'
                  if len(parts) == 2:
                      module_name = parts[1]
                      globals_dict[module_name] = importlib.import_module(module_name)
                      self.import_submodules(module_name, globals_dict)
                  # Handle 'import module as alias'
                  elif len(parts) == 4 and parts[2] == 'as':
                      module_name = parts[1]
                      alias = parts[3]
                      globals_dict[alias] = importlib.import_module(module_name)
                      self.import_submodules(module_name, globals_dict)
              elif imp.startswith('from'):
                  # Handle 'from module import name'
                  if len(parts) == 4:
                      module_name = parts[1]
                      name = parts[3]
                      globals_dict[name] = importlib.import_module(f"{module_name}.{name}")
                  # Handle 'from module import name as alias'
                  elif len(parts) == 6 and parts[4] == 'as':
                      module_name = parts[1]
                      name = parts[3]
                      alias = parts[5]
                      globals_dict[alias] = importlib.import_module(f"{module_name}.{name}")
          except ImportError as e:
              print(f"Failed to import {imp}: {e}")

  def import_submodules(self, package_name, globals_dict):
      """Get the submodules from a package and import those into the globals"""
      if package_name in sys.modules:
          package = sys.modules[package_name]
          if hasattr(package, '__path__'):
              for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
                  if any(submodule.startswith(module_name) for submodule in self.imports):
                      try:
                          module = importlib.import_module(module_name)
                          globals_dict[module_name] = module
                      except ImportError as e:
                          print(f"Failed to import submodule {module_name}: {e}")
                          traceback.print_exc()

  def safe_exec(self, code_string, globals_dict=None, locals_dict=None):
      """Execute """
      if globals_dict is None:
          globals_dict = {}
      if locals_dict is None:
          locals_dict = {}
          
      # Import submodules for each module in globals_dict
      for module_name in list(globals_dict.keys()):
          try:
              self.import_submodules(module_name, globals_dict)
          except Exception as e:
              print(f"Failed to import submodules for {module_name}: {e}")
          
      try:
          exec(sanitize_code(code_string), globals_dict, locals_dict)
      except Exception as e:
          print("An error occurred:")
          traceback.print_exc()
          raise e

  def go(self, prompt:str, any=None, **kwargs):
      """Takes the prompt and inputs, Generates a function with an LLM for the Node"""
      if prompt == "": # if empty, reset
          self.reset()
          return (any,)
      result = None
      if not is_none(any):
          registry = self.FUNCTION_REGISTRY
          print(f"Last Error: {self.last_error}")
          fr = registry.get_function(prompt)
          use_function = fr is not None and self.last_error is None
          use_generation = self.script is None or self.last_prompt != prompt or self.last_error is not None
          if use_generation and not use_function:
              print("Generating Node function...")
              # Generate the function code using OpenAI
              r = self.get_llm_response(prompt, any=any, **kwargs)
              
              # Store the script for future use
              self.script = self.extract_imports(r)
              print(f"Stored script:\n{self.script}")
          if use_function:
              self.script = fr['function']
              self.last_comment = fr['comment']
              self.imports = fr['imports']
          self.last_prompt = prompt

          # Execute the stored script to define the function
          try:
              # Define a dictionary to store globals and locals, updating it with imported libs from script and built in functions
              globals_dict = {"__builtins__": __builtins__}
              # globals_dict.update({imp.split()[1]: globals()[imp.split()[1]] for imp in self.imports if imp.startswith('import')})
              # globals_dict.update({imp.split()[1]: globals()[imp.split()[3]] for imp in self.imports if imp.startswith('from')})
              self._prepare_globals(globals_dict)
              globals_dict.update({"np": np})
              locals_dict = {}

              self.safe_exec(self.script, globals_dict, locals_dict)
          except Exception as e:
              print("--- Exception During Exec ---")
              # store the error for next run
              self.last_error = traceback.format_exc()
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
      # Here we assume the function is complete and we can store it in the registry
      registry.add_function(prompt, self.script, self.imports, self.last_comment, [variable_info(any)])
      return (result,)
  
class AnyNodeGemini(AnyNode):
    def __init__(self, api_key=None):
        super().__init__()
        self.llm = GoogleGemini()

    @classmethod
    def INPUT_TYPES(self):  # pylint: disable = invalid-name, missing-function-docstring
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Take the input and multiply by 5",
                }),
                "model": ("STRING", {
                    "default": "gemini-1.5-flash"
                }),
            },
            "optional": {
                "any": (any_type,),
            },
        }

    def get_response(self, system_message, prompt, model, any=None, **kwargs):
        self.llm.model = model
        return self.llm.get_response(system_message, prompt, any=any)

class AnyNodeOpenAICompatible(AnyNode):
    def __init__(self):
        super().__init__()
        self.llm = OpenAICompatible()

    @classmethod
    def INPUT_TYPES(self):  # pylint: disable = invalid-name, missing-function-docstring
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Take the input and multiply by 5",
                }),
                "model": ("STRING", {
                    "default": "mistral"
                }),
                "server": ("STRING", {
                    "default": "http://localhost:11434"
                }),
            },
            "optional": {
                "any": (any_type,),
                "api_key": ("STRING", {
                    "default": "ollama"
                }),
            },
        }

    def get_response(self, system_message, prompt, server=None, model=None, api_key=None, any=None):
        self.llm.api_key = self.llm.api_key or api_key
        self.llm.model = model or self.llm.model
        self.llm.set_api_server(server)
        return self.llm.get_response(system_message, prompt, any=any)

    
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
