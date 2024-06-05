"""
AnyNode Export Module
---
The idea is to refactor the AnyNode function into a node package
"""

# Used to generate the full class code with integrated AnyNode function
EXPORT_SYSTEM_CLASS = """
# Creating a ComfyUI Node Class
[[CONTROLINFO]]
"""

# Used to rewrite the system maps in the main nodes.py file
EXPORT_SYSTEM_MAPS = """
# Register the Node in the Package
"""

class AnyNodeExport:
    """Exports your AnyNode so it can be a "real boy"!"""
    """
        Needs the system prompt for exporting.
        Copy:

        Needs to create:
            - requirements.txt
            - write the class
            - insert class into it's own python file
            - prepared node class mappings map entries and insert into main nodes.py
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control": ("CTRL", {"forceInput": True}),
                "package": ("STRING", {"default": "TestPack"}),
                "name": ("STRING", {"default": "TestNode"}),
                "save_node": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "comment": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("CTRL",)
    RETURN_NAMES = ("control",)
    FUNCTION = "export"
    OUTPUT_NODE = True

    CATEGORY = "utils"
    
    def copy_core_files(self):
        pass
    
    def package_init(self, control, package_path, name, comment=None):
        self.copy_core_files()

    def export(self, control, package=None, name=None, comment=None, save_node=False, prompt=None, unique_id=None, extra_pnginfo=None):
        """Perform export if save_node is activated"""
        if save_node:
            package_path = f"custom_nodes/{package}/"
            self.package_init(control, package_path, name, comment=comment)
        
        return (control, )
