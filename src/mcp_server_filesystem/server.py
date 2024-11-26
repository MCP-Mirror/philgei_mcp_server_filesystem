import asyncio
import logging
from logging.handlers import RotatingFileHandler
from contextlib import closing
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
from pydantic import AnyUrl

logger = logging.getLogger('mcp_filesystem_server')
logger.info("Starting MCP Filesystem Server")

PROMPT_TEMPLATE = """
Welcome to the Filesystem MCP Server demo! This server allows secure access to your filesystem through the Model Context Protocol.

Current directory access: {allowed_dirs}

Available operations:
- Read files
- Write files
- Create directories
- List directory contents
- Move/rename files
- Search for files
- Get file metadata
- Append to file

Let me know what you'd like to do and I'll help guide you through it!
"""

class FilesystemManager:
    def __init__(self, allowed_dirs: List[str]):
        self.allowed_dirs = [str(Path(d).expanduser().resolve()) for d in allowed_dirs]
        self._validate_directories()
        
    def _validate_directories(self):
        """Validate that all allowed directories exist and are accessible"""
        for dir_path in self.allowed_dirs:
            if not os.path.isdir(dir_path):
                raise ValueError(f"Directory does not exist or is not accessible: {dir_path}")
            if not os.access(dir_path, os.R_OK):
                raise ValueError(f"Directory is not readable: {dir_path}")

    def _normalize_path(self, p: str) -> str:
        """Normalize a path for consistent comparison"""
        return str(Path(p).expanduser().resolve())

    def _validate_path(self, path: str) -> str:
        """Validate that a path is within allowed directories"""
        norm_path = self._normalize_path(path)
        
        # Check if path is within allowed directories
        if not any(norm_path.startswith(allowed_dir) for allowed_dir in self.allowed_dirs):
            raise ValueError(f"Access denied - path outside allowed directories: {path}")
            
        return norm_path

    async def read_file(self, path: str) -> str:
        """Read contents of a file"""
        valid_path = self._validate_path(path)
        try:
            with open(valid_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            raise

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file"""
        valid_path = self._validate_path(path)
        try:
            with open(valid_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Error writing file {path}: {e}")
            raise

    async def create_directory(self, path: str) -> None:
        """Create a new directory"""
        valid_path = self._validate_path(path)
        try:
            os.makedirs(valid_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory {path}: {e}")
            raise

    async def list_directory(self, path: str) -> List[str]:
        """List contents of a directory"""
        valid_path = self._validate_path(path)
        try:
            entries = []
            with os.scandir(valid_path) as it:
                for entry in it:
                    prefix = "[DIR]" if entry.is_dir() else "[FILE]"
                    entries.append(f"{prefix} {entry.name}")
            return entries
        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")
            raise

    async def move_file(self, source: str, destination: str) -> None:
        """Move or rename a file/directory"""
        valid_source = self._validate_path(source)
        valid_dest = self._validate_path(destination)
        try:
            os.rename(valid_source, valid_dest)
        except Exception as e:
            logger.error(f"Error moving {source} to {destination}: {e}")
            raise

    async def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get file/directory metadata"""
        valid_path = self._validate_path(path)
        try:
            stats = os.stat(valid_path)
            return {
                "size": stats.st_size,
                "created": datetime.fromtimestamp(stats.st_ctime),
                "modified": datetime.fromtimestamp(stats.st_mtime),
                "accessed": datetime.fromtimestamp(stats.st_atime),
                "is_directory": os.path.isdir(valid_path),
                "is_file": os.path.isfile(valid_path),
                "permissions": oct(stats.st_mode)[-3:]
            }
        except Exception as e:
            logger.error(f"Error getting file info for {path}: {e}")
            raise

    async def search_files(self, start_path: str, pattern: str) -> List[str]:
        """Search for files matching a pattern"""
        valid_start = self._validate_path(start_path)
        results = []
        
        try:
            for root, _, files in os.walk(valid_start):
                for name in files:
                    if pattern.lower() in name.lower():
                        full_path = os.path.join(root, name)
                        try:
                            self._validate_path(full_path)
                            results.append(full_path)
                        except ValueError:
                            continue
            return results
        except Exception as e:
            logger.error(f"Error searching files from {start_path}: {e}")
            raise
        
    async def append_to_file(self, path: str, content: str) -> None:
        """Append content to a file"""
        valid_path = self._validate_path(path)
        try:
            with open(valid_path, 'a', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Error appending to file {path}: {e}")
            raise

async def main(allowed_dirs: List[str]):
    """Main entry point for the MCP Filesystem Server"""
    logger.info(f"Starting Filesystem MCP Server with allowed directories: {allowed_dirs}")

    fs = FilesystemManager(allowed_dirs)
    server = Server("filesystem-manager")

    @server.list_prompts()
    async def handle_list_prompts() -> List[types.Prompt]:
        logger.debug("Handling list_prompts request")
        return [
            types.Prompt(
                name="fs-demo",
                description="A prompt to demonstrate filesystem operations through MCP",
                arguments=[
                    types.PromptArgument(
                        name="allowed_dirs",
                        description="Directories allowed for access",
                        required=True,
                    )
                ],
            )
        ]

    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: Dict[str, str] | None) -> types.GetPromptResult:
        logger.debug(f"Handling get_prompt request for {name} with args {arguments}")
        if name != "fs-demo":
            raise ValueError(f"Unknown prompt: {name}")

        if not arguments or "allowed_dirs" not in arguments:
            raise ValueError("Missing required argument: allowed_dirs")

        prompt = PROMPT_TEMPLATE.format(allowed_dirs=arguments["allowed_dirs"])

        return types.GetPromptResult(
            description="Filesystem operations demo",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=prompt.strip()),
                )
            ],
        )

    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        """List available filesystem tools"""
        return [
            types.Tool(
                name="read_file",
                description="Read contents of a file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read"},
                    },
                    "required": ["path"],
                },
            ),
            types.Tool(
                name="write_file",
                description="Write content to a file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to write the file"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["path", "content"],
                },
            ),
            types.Tool(
                name="create_directory",
                description="Create a new directory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to create"},
                    },
                    "required": ["path"],
                },
            ),
            types.Tool(
                name="list_directory",
                description="List directory contents",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to list"},
                    },
                    "required": ["path"],
                },
            ),
            types.Tool(
                name="move_file",
                description="Move or rename files and directories",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Source path"},
                        "destination": {"type": "string", "description": "Destination path"},
                    },
                    "required": ["source", "destination"],
                },
            ),
            types.Tool(
                name="search_files",
                description="Search for files matching a pattern",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Starting directory"},
                        "pattern": {"type": "string", "description": "Search pattern"},
                    },
                    "required": ["path", "pattern"],
                },
            ),
            types.Tool(
                name="get_file_info",
                description="Get file/directory metadata",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to inspect"},
                    },
                    "required": ["path"],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: Dict[str, Any] | None
    ) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Handle tool execution requests"""
        try:
            if not arguments:
                raise ValueError("Missing arguments")

            if name == "read_file":
                content = await fs.read_file(arguments["path"])
                return [types.TextContent(type="text", text=content)]

            elif name == "write_file":
                await fs.write_file(arguments["path"], arguments["content"])
                return [types.TextContent(type="text", text=f"Successfully wrote to {arguments['path']}")]

            elif name == "create_directory":
                await fs.create_directory(arguments["path"])
                return [types.TextContent(type="text", text=f"Successfully created directory {arguments['path']}")]

            elif name == "list_directory":
                entries = await fs.list_directory(arguments["path"])
                return [types.TextContent(type="text", text="\n".join(entries))]

            elif name == "move_file":
                await fs.move_file(arguments["source"], arguments["destination"])
                return [types.TextContent(type="text", text=f"Successfully moved {arguments['source']} to {arguments['destination']}")]

            elif name == "search_files":
                results = await fs.search_files(arguments["path"], arguments["pattern"])
                return [types.TextContent(type="text", text="\n".join(results) if results else "No matches found")]

            elif name == "get_file_info":
                info = await fs.get_file_info(arguments["path"])
                return [types.TextContent(type="text", text=str(info))]
            
            elif name == "append_to_file":
                await fs.append_to_file(arguments["path"], arguments["content"])
                return [types.TextContent(type="text", text=f"Successfully appended to {arguments['path']}")]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            logger.error(f"Error in tool {name}: {e}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("Server running with stdio transport")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="filesystem",
                server_version="0.5.1",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python server.py <allowed-directory> [additional-directories...]", file=sys.stderr)
        sys.exit(1)
        
    asyncio.run(main(sys.argv[1:]))