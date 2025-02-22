import json
import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""

    command: str
    args: list[str]
    env: Optional[Dict[str, str]] = None


class MCPConfigManager:
    """Manages MCP server configurations"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize MCP config manager

        Args:
            config_path: Path to MCP config file. If None, will look in default locations
        """
        self.config_path = config_path
        self.servers: Dict[str, MCPServerConfig] = {}
        self._load_config()

    def _load_config(self):
        """Load MCP configuration from file"""
        paths_to_try = [
            self.config_path,
            os.path.join(os.getcwd(), "mcp.json"),
            os.path.join(os.getcwd(), ".mcp.json"),
            os.path.expanduser("~/.mcp.json"),
        ]

        config_file = None
        for path in paths_to_try:
            if path and os.path.exists(path):
                config_file = path
                break

        if not config_file:
            return

        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            if "mcpServers" in config:
                for server_name, server_config in config["mcpServers"].items():
                    self.servers[server_name] = MCPServerConfig(
                        command=server_config["command"],
                        args=server_config["args"],
                        env=server_config.get("env"),
                    )
        except Exception as e:
            raise ValueError(f"Failed to load MCP config from {config_file}: {str(e)}")

    def get_server_config(self, server_name: str) -> MCPServerConfig:
        """Get configuration for a specific MCP server

        Args:
            server_name: Name of the server to get config for

        Returns:
            MCPServerConfig for the specified server

        Raises:
            ValueError: If server_name not found in config
        """
        if server_name not in self.servers:
            raise ValueError(f"MCP server '{server_name}' not found in config")
        return self.servers[server_name]

    def list_servers(self) -> list[str]:
        """Get list of configured server names"""
        return list(self.servers.keys())
