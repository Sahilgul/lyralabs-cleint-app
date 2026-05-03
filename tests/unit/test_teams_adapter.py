"""lyra_core.channels.teams.adapter."""

from __future__ import annotations

import sys

import pytest
from lyra_core.channels.teams.adapter import build_teams_app


def test_build_teams_app_raises_without_botbuilder(monkeypatch) -> None:
    """When optional `botbuilder` extra is missing, raise a friendly RuntimeError."""
    # Force the import inside `build_teams_app` to fail by removing module
    for name in list(sys.modules):
        if name.startswith("botbuilder"):
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = __import__

    def fake_import(name, *args, **kw):
        if name.startswith("botbuilder"):
            raise ImportError("simulated absence")
        return real_import(name, *args, **kw)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="botbuilder-python not installed"):
        build_teams_app()
