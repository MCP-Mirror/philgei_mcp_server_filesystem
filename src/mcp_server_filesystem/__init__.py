from . import server
import asyncio
import argparse
import os

def main():
    """Main entry point for the package."""
    parser = argparse.ArgumentParser(description='Filesystem MCP Server')
    parser.add_argument('allowed_dirs', 
                       nargs='+',
                       help='Space-separated list of directories to allow access to')
    
    args = parser.parse_args()
    
    # Validate and expand paths
    allowed_dirs = [os.path.abspath(os.path.expanduser(d)) for d in args.allowed_dirs]
    
    # Run the server
    asyncio.run(server.main(allowed_dirs))

# Expose important items at package level
__all__ = ["main", "server"]