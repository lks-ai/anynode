"""
@author: newsbubbles
@title: AnyNode
@nickname: anynode
@description: LLM based nodes 
"""

from .nodes.any import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, AnyNode
from .nodes.export import AnyNodeExport

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# from .nodes.any import AnyNode