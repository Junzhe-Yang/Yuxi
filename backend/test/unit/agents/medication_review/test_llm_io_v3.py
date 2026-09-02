from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from yuxi.agents.buildin.medication_review.llm_io import (
    StructuredOutputError,
    invoke_json_schema,
)
from yuxi.agents.buildin.medication_review.models import ReviewAgendaDraft


@pytest.mark.asyncio
async def test_plain_gateway_receives_explicit_schema_and_repairs_format_once():
    prompts: list[str] = []

    class PlainGatewayModel:
        async def ainvoke(self, messages):
            prompts.append(messages[-1].content)
            if len(prompts) == 1:
                return AIMessage(content="not-json")
            return AIMessage(content='{"questions":[]}')

    result = await invoke_json_schema(
        model=PlainGatewayModel(),
        stage="agenda",
        system_prompt="system",
        user_prompt="task",
        output_model=ReviewAgendaDraft,
        repair_limit=1,
        retain_raw_output=False,
    )

    assert len(prompts) == 2
    assert all("目标 JSON Schema" in item for item in prompts)
    assert result.audit.repaired
    assert result.audit.raw_output is None
    assert result.audit.repair_raw_output is None


@pytest.mark.asyncio
async def test_provider_failures_use_only_technical_retry_and_skip_schema_repair():
    calls = 0

    class UnavailableModel:
        async def ainvoke(self, _messages):
            nonlocal calls
            calls += 1
            raise RuntimeError("provider unavailable")

    with pytest.raises(StructuredOutputError) as exc_info:
        await invoke_json_schema(
            model=UnavailableModel(),
            stage="agenda",
            system_prompt="system",
            user_prompt="task",
            output_model=ReviewAgendaDraft,
            repair_limit=1,
            technical_retry_limit=1,
        )

    assert calls == 2
    assert exc_info.value.audit.failure_kind == "provider"
    assert exc_info.value.audit.repair_raw_output is None
