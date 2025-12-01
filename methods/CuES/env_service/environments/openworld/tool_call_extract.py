import re
import json


def clean_pseudo_json(text: str) -> str:
    """Normalize loosely formatted JSON-like text into valid JSON string.

    - Remove // and # comments
    - Quote bare keys
    - Replace single quotes with double quotes
    - Remove trailing commas before closing brackets/braces
    """
    # Remove comments
    text = re.sub(r'//.*|#.*', '', text)
    # Add quotes around bare keys
    text = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b\s*:', r'"\1":', text)
    # Replace single quotes with double quotes
    text = text.replace("'", '"')
    # Remove trailing commas
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return text.strip()


def extract_tool_calls(text: str):
    """Extract a list of tool_calls from mixed text.

    Returns a list of dicts with keys: tool_name, tool_args.
    Strategy:
      1) Prefer parsing fenced ```json ...``` blocks containing a list.
      2) Fallback to bracketed blocks that contain 'tool_name' and try to clean/parse.
    """
    tool_calls = []

    # 1) Prefer matching ```json [...]``` code blocks
    json_block_pattern = re.compile(r'```json\s*(\[[\s\S]*?\])\s*```', re.MULTILINE)
    matches = json_block_pattern.findall(text)

    for block in matches:
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                parsed = [parsed]
            for item in parsed:
                if 'tool_name' in item and 'tool_args' in item:
                    tool_calls.append({
                        "tool_name": item["tool_name"],
                        "tool_args": item["tool_args"]
                    })
        except Exception as e:
            print(f"[JSON block parse failed]: {e}\nContent:\n{block}")

    if tool_calls:
        return tool_calls  # Found valid tool_calls; return early

    # 2) Fallback: match any bracketed block containing 'tool_name'
    bracket_pattern = re.compile(r'\[\s*{[\s\S]*?}\s*\]', re.DOTALL)
    candidates = bracket_pattern.findall(text)
    for c in candidates:
        if 'tool_name' not in c:
            continue
        cleaned = clean_pseudo_json(c)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                parsed = [parsed]
            for item in parsed:
                if 'tool_name' in item and 'tool_args' in item:
                    tool_calls.append({
                        "tool_name": item["tool_name"],
                        "tool_args": item["tool_args"]
                    })
        except Exception as e:
            print(f"[Pseudo-JSON parse failed]: {e}\nContent:\n{cleaned}")

    return tool_calls


if __name__ == "__main__":
    sample = (
        "To help decide whether a stock is worth buying, let's first fetch the latest market data.\n\n"
        "```json\n[\n  {\n    \"tool_name\": \"tdx_PBHQInfo_quotes\",\n    \"tool_args\": {\n      \"code\": \"300750\",\n      \"setcode\": \"2\"\n    }\n  }\n]\n```\n\n"
    )
    print(extract_tool_calls(sample))