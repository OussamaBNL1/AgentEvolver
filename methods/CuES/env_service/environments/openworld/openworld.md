1. The MCP servers used by openworld are configured in `mcp_tool.json` (specified in `openworld_env.py`, via default config or a custom path in params).
2. The tool-call format is defined by a custom `system_prompt` in `openworld_env.py`; you can modify this format as needed.
3. There is no built-in query set or evaluation method; configure them in `openworld_env.py` as required.
4. The tool-call parsing logic is implemented in `tool_call_extract.py`; feel free to customize it.