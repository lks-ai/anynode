"""
@author: newsbubbles
@title: AnyNode
@nickname: anynode
@description: LLM based nodes 
"""

from .nodes.any import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, AnyNode
from .nodes.export import AnyNodeExport
from .nodes.show_code import NODE_CLASS_MAPPINGS as SHOW_CODE_NODE_CLASS_MAPPINGS
from .nodes.show_code import NODE_DISPLAY_NAME_MAPPINGS as SHOW_CODE_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS.update(SHOW_CODE_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SHOW_CODE_NODE_DISPLAY_NAME_MAPPINGS)
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

# from .nodes.any import AnyNode
