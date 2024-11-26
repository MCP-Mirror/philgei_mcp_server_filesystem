# MCP Filesystem Server

> **DISCLAIMER**: This is an unofficial port of Claude's filesystem MCP server implementation, created solely as a Proof of Concept (POC). This project is not affiliated with Anthropic and should not be used in production environments. It was created to demonstrate and explore MCP server functionality.
>
> The original implementation and concept were demonstrated by Claude (Anthropic) during interactive sessions. This port serves educational purposes only.

## Quick Start

1. Install using UVX:
```bash
uv venv
.venv\Scripts\activate  # On Windows
uv pip install -e .
```

## Usage with Claude Desktop

1. Edit your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "myFiles": {
      "command": "mcp-server-filesystem",
      "args": [
        "D:/"  // Replace with your desired directory
      ]
    }
  }
}
```

You can add multiple directories:
```json
{
  "mcpServers": {
    "myFiles": {
      "command": "mcp-server-filesystem",
      "args": [
        "D:/",
        "C:/Users/YourUsername/Documents",
        "~/Desktop"
      ]
    }
  }
}
```

2. Restart Claude Desktop to apply changes

3. In Claude Desktop:
   - Click the paperclip icon
   - Select the MCP menu (two electrical plugs connecting)
   - Your filesystem server will be available in the integrations

## Security Note
Only directories specified in the `args` array will be accessible. The server enforces strict path validation to prevent access outside allowed directories.

## Troubleshooting

1. Command not found:
   - Make sure the package is installed (`uv pip list`)
   - Check your Python environment is activated

2. Access denied:
   - Verify the directories in `args` exist
   - Ensure you have read permissions for specified directories

3. Server not showing in Claude Desktop:
   - Check your config file syntax
   - Restart Claude Desktop after config changes