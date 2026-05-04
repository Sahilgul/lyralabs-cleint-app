"""Slack-native tools (the in-process equivalent of mcp.slack.com).

Importing this module registers every Slack tool on the default registry.
Mirrors and extends the surface of the official Slack MCP server but
talks straight to the Slack Web API using the bot/user tokens captured
at OAuth install time -- skipping the JSON-RPC-over-HTTP transport
(we'd just be calling our own MCP from our own process).

Tool surface (read = LOW, write = MEDIUM unless noted):
  Communication: chat.send_message, chat.schedule_message
  Threading:     conversations.history, conversations.replies
  Discovery:     conversations.list, conversations.info, conversations.open
  Membership:    conversations.invite, conversations.create
  Users:         users.info, users.list, users.lookup_by_email
  Search:        search.messages (user token), search.files (user token)
  Reactions:     reactions.add, reactions.remove           (LOW)
  Curation:      pins.add (LOW), bookmarks.add (LOW)
  Files:         files.upload
  Canvas:        canvas.create, canvas.update, canvas.read
  Reminders:     reminders.add (LOW)
"""

from .bookmarks import BookmarksAdd
from .canvas import CanvasCreate, CanvasRead, CanvasUpdate
from .chat import ChatScheduleMessage, ChatSendMessage
from .conversations import (
    ConversationsCreate,
    ConversationsHistory,
    ConversationsInfo,
    ConversationsInvite,
    ConversationsList,
    ConversationsOpen,
    ConversationsReplies,
)
from .files import FilesUpload
from .pins import PinsAdd
# from .reactions import ReactionsAdd, ReactionsRemove  # disabled 2026-05 — speak, don't react
from .reminders import RemindersAdd
from .search import SearchFiles, SearchMessages
from .users import UsersInfo, UsersList, UsersLookupByEmail

__all__ = [
    "BookmarksAdd",
    "CanvasCreate",
    "CanvasRead",
    "CanvasUpdate",
    "ChatScheduleMessage",
    "ChatSendMessage",
    "ConversationsCreate",
    "ConversationsHistory",
    "ConversationsInfo",
    "ConversationsInvite",
    "ConversationsList",
    "ConversationsOpen",
    "ConversationsReplies",
    "FilesUpload",
    "PinsAdd",
    # "ReactionsAdd",    # disabled 2026-05
    # "ReactionsRemove", # disabled 2026-05
    "RemindersAdd",
    "SearchFiles",
    "SearchMessages",
    "UsersInfo",
    "UsersList",
    "UsersLookupByEmail",
]
