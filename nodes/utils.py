import json
import os
import re
import numpy as np
import torch
from .context_utils import is_context_empty

class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

  def __ne__(self, __value: object) -> bool:
    return False


class ContainsAnyDict(dict):
  """A special class that always returns true for contains check ('prop' in my_dict)."""

  def __contains__(self, key):
    return True


any_type = AnyType("*")


def is_dict_value_falsy(data: dict, dict_key: str):
  """ Checks if a dict value is falsy."""
  val = get_dict_value(data, dict_key)
  return not val


def get_dict_value(data: dict, dict_key: str, default=None):
  """ Gets a deeply nested value given a dot-delimited key."""
  keys = dict_key.split('.')
  key = keys.pop(0) if len(keys) > 0 else None
  found = data[key] if key in data else None
  if found is not None and len(keys) > 0:
    return get_dict_value(found, '.'.join(keys), default)
  return found if found is not None else default


def set_dict_value(data: dict, dict_key: str, value, create_missing_objects=True):
  """ Sets a deeply nested value given a dot-delimited key."""
  keys = dict_key.split('.')
  key = keys.pop(0) if len(keys) > 0 else None
  if key not in data:
    if create_missing_objects == False:
      return
    data[key] = {}
  if len(keys) == 0:
    data[key] = value
  else:
    set_dict_value(data[key], '.'.join(keys), value, create_missing_objects)

  return data


def dict_has_key(data: dict, dict_key):
  """ Checks if a dict has a deeply nested dot-delimited key."""
  keys = dict_key.split('.')
  key = keys.pop(0) if len(keys) > 0 else None
  if key is None or key not in data:
    return False
  if len(keys) == 0:
    return True
  return dict_has_key(data[key], '.'.join(keys))


def load_json_file(file: str, default=None):
  """Reads a json file and returns the json dict, stripping out "//" comments first."""
  if path_exists(file):
    with open(file, 'r', encoding='UTF-8') as file:
      config = re.sub(r"(?:^|\s)//.*", "", file.read(), flags=re.MULTILINE)
    return json.loads(config)
  return default


def save_json_file(file_path: str, data: dict):
  """Saves a json file."""
  os.makedirs(os.path.dirname(file_path), exist_ok=True)
  with open(file_path, 'w+', encoding='UTF-8') as file:
    json.dump(data, file, sort_keys=False, indent=2, separators=(",", ": "))

def path_exists(path):
  """Checks if a path exists, accepting None type."""
  if path is not None:
    return os.path.exists(path)
  return False

def is_none(value):
    """Checks if a value is none. Pulled out in case we want to expand what 'None' means."""
    if value is not None:
        if isinstance(value, dict) and 'model' in value and 'clip' in value:
            return is_context_empty(value)
    return value is None

def get_variable_info(variable):
    info = {}
    
    # Get the exact type of the variable
    #info['type'] = type(variable).__name__
    info['type'] = str(type(variable))

    # Check for common types and add relevant information
    if isinstance(variable, (np.ndarray, torch.Tensor)):
        info['shape'] = tuple(variable.shape)
        info['dtype'] = str(variable.dtype)
    elif isinstance(variable, (list, tuple, set)):
        info['length'] = len(variable)
        if len(variable) > 128:
            info['structure'] = f"{info['type']} of length {info['length']}"
        else:
            info['structure'] = [type(v).__name__ for v in variable]
    elif isinstance(variable, dict):
        info['length'] = len(variable)
        if len(variable) > 128:
            info['structure'] = f"{info['type']} of length {info['length']}"
        else:
            info['keys'] = list(variable.keys())
            info['values'] = [type(variable[k]).__name__ for k in variable]
    elif isinstance(variable, str):
        info['length'] = len(variable)
    elif hasattr(variable, '__dict__'):
        # For user-defined classes, show their attributes
        info['attributes'] = vars(variable)

    return info

def variable_info(variable):
    info = get_variable_info(variable)
    info_str = f"Type: {info['type']}"
    
    if 'shape' in info:
        info_str += f", Shape: {info['shape']}, Dtype: {info['dtype']}"
    if 'length' in info:
        info_str += f", Length: {info['length']}"
    if 'structure' in info:
        info_str += f", Structure: {info['structure']}"
    if 'keys' in info:
        info_str += f", Keys: {info['keys']}"
    if 'values' in info:
        info_str += f", Values: {info['values']}"
    if 'attributes' in info:
        info_str += f", Attributes: {info['attributes']}"

    return info_str

import ast

class CodeSanitizer(ast.NodeVisitor):
    def __init__(self):
        self.dangerous_constructs = [
            'eval', 'exec', 'input', '__import__', 'os', 'subprocess', 'shutil', 'sys', 'compile'
        ]
        self.safe_constructs = {
            'open': [('wave', 'io.BytesIO')],
            'Image.open': [('PIL.Image', 'BytesIO')],
        }
        self.errors = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.dangerous_constructs:
            self.errors.append(f"Dangerous construct detected. You cannot use: {node.func.id}")
        elif isinstance(node.func, ast.Attribute) and node.func.attr in self.dangerous_constructs:
            self.errors.append(f"Dangerous construct detected. You cannot use: {node.func.attr}")
        elif isinstance(node.func, ast.Name) and node.func.id == 'open':
            if not self.is_safe_open(node):
                self.errors.append(f"Dangerous construct detected. You cannot use: {node.func.id}")
        elif isinstance(node.func, ast.Attribute) and node.func.attr == 'open':
            if not self.is_safe_open(node):
                self.errors.append(f"Dangerous construct detected. You cannot use: {node.func.attr}")

        self.generic_visit(node)

    def is_safe_open(self, node):
        for arg in node.args:
            if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute):
                module = arg.func.value.id if isinstance(arg.func.value, ast.Name) else None
                construct = arg.func.attr
                if (module, construct) in self.safe_constructs.get('open', []):
                    return True
                if isinstance(arg.func, ast.Name) and arg.func.id == 'Image' and 'open' in [x[1] for x in self.safe_constructs['Image.open']]:
                    return True
        return False

    def visit_While(self, node):
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            self.errors.append("Potential infinite loop detected: while True")
        self.generic_visit(node)

    def visit_For(self, node):
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'iter':
            self.errors.append("Potential infinite loop detected: for ... in iter(...)")
        self.generic_visit(node)

def sanitize_code(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in code: {e}")

    sanitizer = CodeSanitizer()
    sanitizer.visit(tree)

    if sanitizer.errors:
        raise ValueError('\n'.join(sanitizer.errors))

    return code
