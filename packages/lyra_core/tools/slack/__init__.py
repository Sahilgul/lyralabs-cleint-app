"""Slack-native tools (the in-process equivalent of mcp.slack.com).

Importing this module registers every Slack tool on the default registry.
Mirrors the surface of the official Slack MCP server (conversations,
users, search, canvas) but talks straight to the Slack Web API using
the bot/user tokens captured at OAuth install time -- skipping the
JSON-RPC-over-HTTP transport (we'd just be calling our own MCP from
our own process).
"""

from .canvas import CanvasCreate
from .conversations import ConversationsHistory, ConversationsReplies
from .search import SearchMessages
from .users import UsersInfo, UsersList

__all__ = [
    "ConversationsHistory",
    "ConversationsReplies",
    "UsersInfo",
    "UsersList",
    "SearchMessages",
    "CanvasCreate",
]
