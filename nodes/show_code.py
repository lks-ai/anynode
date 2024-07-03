from typing import Optional

from server import PromptServer
import json


class AnyNodeShowCode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "control": ("CTRL", {}),
                "show": (["code", "response"], {"default": "code"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    CATEGORY = "utils"
    COLOR = "#FFA800"
    DESCRIPTION = "Shows code generated or json response from any node"

    OUTPUT_NODE = True
    RETURN_TYPES = ()

    FUNCTION = "__call__"

    @staticmethod
    def send_code(code: str, control: dict, language: str, unique_id: int):
        event = dict(
            code=code,
            control=control,
            language=language,
            unique_id=unique_id
        )
        PromptServer.instance.send_sync(f"any-node-show-code-{unique_id}", event)

    def __call__(self, control: Optional[dict] = None, show: str = "code", unique_id: int = 0):
        code = "# Waiting for code..."
        language = "python"

        if control is None:
            control = {"function": "# Waiting for code..."}
            return {"ui": {"code": [code], "control": control, "language": [language], "unique_id": [unique_id]}}

        if show == 'code' or show is None:
            code = control.get("function", "# Waiting for code...")
        else:
            code = json.dumps(control, indent=4)
            language = "json"

        self.send_code(code, control, language, unique_id)
        return {"ui": {"code": [code], "control": [control], "language": [language], "unique_id": [unique_id]}}


NODE_CLASS_MAPPINGS = {"AnyNodeShowCode": AnyNodeShowCode}
NODE_DISPLAY_NAME_MAPPINGS = {"AnyNodeShowCode": "Any Node 🍄 (Show Code)"}
