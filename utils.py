import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_nested(obj):
    if isinstance(obj, str):
        try:
            return _parse_nested(json.loads(obj))
        except (json.JSONDecodeError, ValueError):
            return obj
    elif isinstance(obj, dict):
        return {k: _parse_nested(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_parse_nested(i) for i in obj]
    return obj

def _read_jsonl(path):
    with open(path, "r") as f:
        return [_parse_nested(json.loads(line)) for line in f if line.strip()]

def get_image_file_names(image_path:os.PathLike)-> Tuple[List[Path], List[str], List[str]]:
    image_path = Path(image_path)
    file_paths = sorted(image_path.glob("*.jpg"))
    file_names = [path.name for path in file_paths]
    timestamp_ns = [file_name.split("-")[0] for file_name in file_names]
    return file_paths, file_names, timestamp_ns

def get_hashed_image_jsonl_data(data_path:os.PathLike):
    DATA_PATH = Path(data_path)
    JSONL_PATH = next((DATA_PATH / f"data/").glob("*.jsonl"))
    IMAGE_PATH = DATA_PATH / "images/"

    _, _, image_timestamps_ns = get_image_file_names(IMAGE_PATH)
    jsonl_data = _read_jsonl(JSONL_PATH)

    hashed_timestamp_ns_jsonl_data = {}
    for jsonl_line in jsonl_data:
        try:
            jsonl_timestamp_ns = str(jsonl_line["value"].get("image_timestamp_ns"))
            if jsonl_timestamp_ns in image_timestamps_ns:
                hashed_timestamp_ns_jsonl_data[jsonl_timestamp_ns] = jsonl_line
        except (AttributeError):
            pass

    return hashed_timestamp_ns_jsonl_data