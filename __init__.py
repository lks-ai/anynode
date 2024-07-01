"""
@author: newsbubbles
@title: AnyNode
@nickname: anynode
@description: LLM based nodes 
"""

from .nodes.any import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, WEB_DIRECTORY, AnyNode
from .nodes.export import AnyNodeExport

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]

# from .nodes.any import AnyNode