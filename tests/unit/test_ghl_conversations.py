"""lyra_core.tools.ghl.conversations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import respx
from lyra_core.tools.base import ToolError
from lyra_core.tools.credentials import ProviderCredentials
from lyra_core.tools.ghl.client import GHL_BASE
from lyra_core.tools.ghl.conversations import GhlSendMessage, GhlSendMessageInput


def _creds():
    return ProviderCredentials(
        provider="ghl",
        access_token="t",
        refresh_token="rt",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        external_account_id="loc",
        scopes="conversations.write",
        metadata={"location_id": "loc"},
    )


@pytest.mark.asyncio
async def test_email_requires_subject(make_ctx) -> None:
    ctx = make_ctx(creds=_creds())
    with pytest.raises(ToolError, match="subject is required"):
        await GhlSendMessage().run(
            ctx,
            GhlSendMessageInput(contact_id="c-1", type="Email", message="hi"),
        )


@pytest.mark.asyncio
async def test_message_or_html_required(make_ctx) -> None:
    ctx = make_ctx(creds=_creds())
    with pytest.raises(ToolError, match="message or html"):
        await GhlSendMessage().run(
            ctx,
            GhlSendMessageInput(contact_id="c-1", type="SMS"),
        )


@pytest.mark.asyncio
async def test_sms_dry_run(make_ctx) -> None:
    ctx = make_ctx(creds=_creds(), dry_run=True)
    out = await GhlSendMessage().run(
        ctx,
        GhlSendMessageInput(contact_id="c-1", type="SMS", message="hi there"),
    )
    assert out.status == "dry-run"


@pytest.mark.asyncio
async def test_sms_post(make_ctx) -> None:
    ctx = make_ctx(creds=_creds())
    with respx.mock(base_url=GHL_BASE) as mock:
        route = mock.post("/conversations/messages").respond(
            200,
            json={"conversationId": "cv-1", "messageId": "msg-1", "status": "delivered"},
        )
        out = await GhlSendMessage().run(
            ctx,
            GhlSendMessageInput(contact_id="c-1", type="SMS", message="hello"),
        )

    body = route.calls[0].request.read().decode()
    assert '"type":"SMS"' in body or '"type": "SMS"' in body
    assert '"contactId":"c-1"' in body or '"contactId": "c-1"' in body
    assert "subject" not in body
    assert out.conversation_id == "cv-1"
    assert out.message_id == "msg-1"
    assert out.status == "delivered"


@pytest.mark.asyncio
async def test_email_includes_subject_and_html(make_ctx) -> None:
    ctx = make_ctx(creds=_creds())
    with respx.mock(base_url=GHL_BASE) as mock:
        route = mock.post("/conversations/messages").respond(200, json={})
        await GhlSendMessage().run(
            ctx,
            GhlSendMessageInput(
                contact_id="c-1",
                type="Email",
                subject="Hello!",
                message="text version",
                html="<p>html version</p>",
            ),
        )
    body = route.calls[0].request.read().decode()
    assert '"subject":"Hello!"' in body or '"subject": "Hello!"' in body
    assert "html version" in body


@pytest.mark.asyncio
async def test_default_status_sent_when_response_missing(make_ctx) -> None:
    ctx = make_ctx(creds=_creds())
    with respx.mock(base_url=GHL_BASE) as mock:
        mock.post("/conversations/messages").respond(200, json={})
        out = await GhlSendMessage().run(
            ctx,
            GhlSendMessageInput(contact_id="c-1", type="SMS", message="hi"),
        )
    assert out.status == "sent"


def test_send_message_requires_approval() -> None:
    assert GhlSendMessage.requires_approval is True
