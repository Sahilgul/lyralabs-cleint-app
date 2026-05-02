"""lyra_core.channels.schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lyra_core.channels.schema import Artifact, InboundMessage, OutboundReply, Surface


class TestSurfaceEnum:
    def test_string_values(self) -> None:
        assert Surface.SLACK.value == "slack"
        assert Surface.TEAMS.value == "teams"

    def test_str_subclass(self) -> None:
        assert str(Surface.SLACK) == "slack"


class TestInboundMessage:
    def test_minimal_valid(self) -> None:
        m = InboundMessage(
            surface=Surface.SLACK,
            tenant_external_id="T1",
            channel_id="C1",
            thread_id="123.456",
            agent_thread_id="slack:dm:T1:C1:U1",
            user_id="U1",
            text="hello",
        )
        assert m.surface == Surface.SLACK
        assert m.agent_thread_id == "slack:dm:T1:C1:U1"
        assert m.files == []
        assert m.raw == {}

    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            InboundMessage()  # type: ignore[call-arg]

    def test_agent_thread_id_required(self) -> None:
        with pytest.raises(ValidationError):
            InboundMessage(  # type: ignore[call-arg]
                surface=Surface.SLACK,
                tenant_external_id="T1",
                channel_id="C1",
                thread_id="t",
                user_id="U1",
                text="hi",
            )

    def test_serializes_round_trip(self) -> None:
        m = InboundMessage(
            surface=Surface.SLACK,
            tenant_external_id="T1",
            channel_id="C1",
            thread_id="123.456",
            agent_thread_id="slack:ch:T1:C1:123.456",
            user_id="U1",
            text="hi",
            files=[{"id": "F1", "name": "x.pdf"}],
            raw={"event": "message"},
        )
        j = m.model_dump_json()
        m2 = InboundMessage.model_validate_json(j)
        assert m2 == m

    def test_accepts_string_for_surface(self) -> None:
        m = InboundMessage(
            surface="slack",  # type: ignore[arg-type]
            tenant_external_id="T1",
            channel_id="C1",
            thread_id="t",
            agent_thread_id="slack:dm:T1:C1:U1",
            user_id="U1",
            text="x",
        )
        assert m.surface == Surface.SLACK


class TestArtifact:
    @pytest.mark.parametrize("kind", ["pdf", "png", "csv", "xlsx", "pptx", "json", "md"])
    def test_accepted_kinds(self, kind: str) -> None:
        a = Artifact(kind=kind, filename=f"x.{kind}", content=b"data")
        assert a.kind == kind

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValidationError):
            Artifact(kind="exe", filename="x.exe", content=b"")  # type: ignore[arg-type]

    def test_content_must_be_bytes(self) -> None:
        a = Artifact(kind="pdf", filename="x.pdf", content=b"binary")
        assert isinstance(a.content, bytes)


class TestOutboundReply:
    def test_minimal_valid(self) -> None:
        r = OutboundReply(channel_id="c")
        assert r.text is None
        assert r.blocks is None
        assert r.thread_ts is None
        assert r.artifacts == []
        assert r.requires_approval is False

    def test_top_level_default(self) -> None:
        # No thread_ts means the bot replies as a new top-level message
        # (the natural UX for DMs and avoids the "all replies appear under
        # the user's first message" bug we hit in production).
        r = OutboundReply(text="hi", channel_id="c")
        assert r.thread_ts is None

    def test_threaded_reply(self) -> None:
        r = OutboundReply(text="ok", channel_id="c", thread_ts="123.456")
        assert r.thread_ts == "123.456"

    def test_with_artifacts(self) -> None:
        r = OutboundReply(
            text="here you go",
            channel_id="c",
            thread_ts="t",
            artifacts=[Artifact(kind="pdf", filename="r.pdf", content=b"%PDF")],
        )
        assert len(r.artifacts) == 1
        assert r.artifacts[0].filename == "r.pdf"

    def test_approval_payload_optional(self) -> None:
        r = OutboundReply(
            channel_id="c",
            thread_ts="t",
            requires_approval=True,
            approval_payload={"job_id": "j-1"},
        )
        assert r.requires_approval is True
        assert r.approval_payload == {"job_id": "j-1"}
