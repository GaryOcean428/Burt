import re
import regex
from typing import Any
from .dirty_json import DirtyJson


def json_parse_dirty(json: str) -> dict[str, Any] | None:
    ext_json = extract_json_object_string(json)
    if ext_json:
        data = DirtyJson.parse_string(ext_json)
        if isinstance(data, dict):
            return data
    return None


def extract_json_object_string(content):
    start = content.find("{")
    if start == -1:
        return ""

    end = content.rfind("}")
    if end == -1:
        return content[start:]
    else:
        return content[start : end + 1]


def extract_json_string(content):
    pattern = r'\{(?:[^{}]|(?R))*\}|\[(?:[^\[\]]|(?R))*\]|"(?:\\.|[^"\\])*"|true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
    match = regex.search(pattern, content)

    if match:
        return match.group(0)
    else:
        print("No JSON content found.")
        return ""


def fix_json_string(json_string):
    def replace_unescaped_newlines(match):
        return match.group(0).replace("\n", "\\n")

    fixed_string = re.sub(
        r'(?<=: ")(.*?)(?=")', replace_unescaped_newlines, json_string, flags=re.DOTALL
    )
    return fixed_string
