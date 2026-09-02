import os

import pytest

pytest.importorskip("langsmith")

from telemetry import configure_langsmith_tracing  # noqa: E402


def test_configure_langsmith_tracing_sets_default_env_vars(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

    configure_langsmith_tracing()

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "clueforagents"


def test_configure_langsmith_tracing_respects_custom_project_name(monkeypatch):
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

    configure_langsmith_tracing(project_name="my-experiment")

    assert os.environ["LANGSMITH_PROJECT"] == "my-experiment"


def test_configure_langsmith_tracing_does_not_override_existing_env_vars(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGSMITH_PROJECT", "already-set")

    configure_langsmith_tracing()

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGSMITH_PROJECT"] == "already-set"
