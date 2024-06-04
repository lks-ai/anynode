class AnyNodeExport:
    """Exports your AnyNode so it can be a real boy!"""
    """
        Needs the system prompt for exporting.
        Copy:

        Needs to create:
            - requirements.txt
            - write the class
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control": ("CTRL", {"forceInput": True}),
                "package": ("STRING", {"default": "DefaultPack"}),
                "name": ("STRING", {}),
                "save_node": ("BOOLEAN", {"default": True}),
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

    #INPUT_IS_LIST = True
    RETURN_TYPES = ("CTRL",)
    RETURN_NAMES = ("control",)
    FUNCTION = "export"
    OUTPUT_NODE = True
    #OUTPUT_IS_LIST = (True,)

    CATEGORY = "utils"

    def export(self, control, text=None, unique_id=None, extra_pnginfo=None):
        pass
