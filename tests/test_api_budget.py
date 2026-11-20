"""Offline checks: no free-model quota is consumed by these tests."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from assignment import pipeline
from agents import agent as agent_module
from core import openai_runtime, utils
from assignment import audit_log, monitoring


def test_blue_disables_implicit_retries():
    runner = openai_runtime.OpenAIRunner(app_name="test", model="test", provider="openrouter")
    with patch("openai.OpenAI") as client:
        first = runner._client()
        assert runner._client() is first
        assert client.call_count == 1
        runner.close()
        first.close.assert_called_once()
    assert client.call_args.kwargs["max_retries"] == 0
    assert client.call_args.kwargs["timeout"] == 45.0


@pytest.mark.parametrize("fail", [False, True])
def test_suite_budget_and_provider_failure(tmp_path, monkeypatch, fail):
    plugins = pipeline.build_production_plugins()
    audit, monitor = pipeline.build_observability()
    runner = openai_runtime.OpenAIRunner(app_name="test", model="test", plugins=plugins, provider="openrouter")
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Banking help."))])
    create = Mock(return_value=completion)
    if fail:
        create.side_effect = RuntimeError("simulated provider failure")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(runner, "_client", lambda: client)
    monkeypatch.setattr(agent_module, "create_blue_agent", lambda _: (SimpleNamespace(instruction="test"), runner))
    monkeypatch.setattr(pipeline, "__file__", str(tmp_path / "src" / "assignment" / "pipeline.py"))
    monkeypatch.setattr(audit_log, "default_audit_log_path", lambda: str(tmp_path / "outputs" / "audit_log.json"))
    monkeypatch.setattr(monitoring, "default_metrics_path", lambda: str(tmp_path / "outputs" / "metrics.json"))
    # Advance the local limiter clock during waits, without sleeping or networking.
    clock = [1000.0]
    async def sleep(seconds):
        clock[0] += seconds
    monkeypatch.setattr(pipeline.time, "time", lambda: clock[0])
    monkeypatch.setattr(pipeline.asyncio, "sleep", sleep)
    async def chat(agent, runner, text):
        return await runner.chat(agent, text), None
    monkeypatch.setattr(utils, "chat_with_agent", chat)
    args = {"plugins": plugins, "audit": audit, "monitor": monitor}
    if fail:
        with pytest.raises(RuntimeError):
            asyncio.run(pipeline.run_assignment_suite(args))
        assert create.call_count == 1
        assert not (tmp_path / "outputs" / "results.json").exists()
        assert json.loads((tmp_path / "outputs" / "suite_error.json").read_text())["completed"] is False
        assert audit.logs[-1]["layer"] == "api_error"
    else:
        result = asyncio.run(pipeline.run_assignment_suite(args))
        assert create.call_count == 6  # Five safe queries, one benign external document.
        assert all(call.kwargs["max_tokens"] == 1024 for call in create.call_args_list)
        assert result["rate_limit"]["passed"] == 10
        assert result["rate_limit"]["blocked"] == 2
        assert not any(q["blocked"] for q in result["safe_queries"])
        assert len(audit.logs) == monitor.total_requests == 28
        import jsonschema
        root = Path(__file__).resolve().parents[1]
        jsonschema.validate(result, json.loads((root / "schemas" / "results.schema.json").read_text()))
    assert (tmp_path / "outputs" / "audit_log.json").exists()
    assert (tmp_path / "outputs" / "metrics.json").exists()
