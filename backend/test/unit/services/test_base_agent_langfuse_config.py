from __future__ import annotations

from types import SimpleNamespace

import pytest

from yuxi.agents.base import BaseAgent


class _FakeGraph:
    def __init__(self):
        self.last_stream_config = None
        self.last_invoke_config = None

    async def astream(self, payload, *, stream_mode, context, config):
        self.last_stream_config = config
        if isinstance(stream_mode, list):
            yield "values", {"action_directive": "action"}
            yield (
                "messages",
                (
                    SimpleNamespace(content="raw", model_dump=lambda: {"type": "ai"}),
                    {"node": "llm"},
                ),
            )
            return
        yield SimpleNamespace(model_dump=lambda: {"type": "ai"}), {"node": "llm"}

    async def ainvoke(self, payload, *, context, config):
        self.last_invoke_config = config
        return {"messages": []}


class _TestAgent(BaseAgent):
    name = "test_agent"
    description = "test"

    async def get_graph(self, **kwargs):
        if getattr(self, "_graph", None) is None:
            self._graph = _FakeGraph()
        return self._graph


_TestAgent.__module__ = "yuxi.agents.tests.fake"


class _ProjectingAgent(_TestAgent):
    def project_stream_message(self, message, state_values, *, context=None):
        return SimpleNamespace(
            content=f"{message.content}-projected",
            state_values=state_values,
            context=context,
        )


_ProjectingAgent.__module__ = "yuxi.agents.tests.fake"


@pytest.mark.asyncio
async def test_base_agent_stream_messages_passes_callbacks_metadata_and_tags():
    agent = _TestAgent()

    items = []
    async for item in agent.stream_messages(
        ["hello"],
        input_context={"user_id": "user-1", "thread_id": "thread-1"},
        callbacks=["handler-1"],
        metadata={"langfuse_user_id": "user-1"},
        tags=["yuxi"],
    ):
        items.append(item)

    graph = await agent.get_graph()
    assert len(items) == 1
    assert graph.last_stream_config == {
        "configurable": {"thread_id": "thread-1", "user_id": "user-1"},
        "recursion_limit": 300,
        "callbacks": ["handler-1"],
        "metadata": {"langfuse_user_id": "user-1"},
        "tags": ["yuxi"],
    }


@pytest.mark.asyncio
async def test_base_agent_invoke_messages_passes_callbacks_metadata_and_tags():
    agent = _TestAgent()

    await agent.invoke_messages(
        ["hello"],
        input_context={"user_id": "user-1", "thread_id": "thread-1"},
        callbacks=["handler-1"],
        metadata={"langfuse_user_id": "user-1"},
        tags=["yuxi"],
    )

    graph = await agent.get_graph()
    assert graph.last_invoke_config == {
        "configurable": {"thread_id": "thread-1", "user_id": "user-1"},
        "recursion_limit": 100,
        "callbacks": ["handler-1"],
        "metadata": {"langfuse_user_id": "user-1"},
        "tags": ["yuxi"],
    }


@pytest.mark.asyncio
async def test_base_agent_projects_message_stream_with_latest_state_and_context():
    agent = _ProjectingAgent()

    items = [
        item
        async for item in agent.stream_messages_with_state(
            ["hello"],
            input_context={"user_id": "user-1", "thread_id": "thread-1"},
        )
    ]

    assert items[0] == ("values", {"action_directive": "action"})
    message, metadata = items[1][1]
    assert message.content == "raw-projected"
    assert message.state_values == {"action_directive": "action"}
    assert message.context.user_id == "user-1"
    assert metadata == {"node": "llm"}
