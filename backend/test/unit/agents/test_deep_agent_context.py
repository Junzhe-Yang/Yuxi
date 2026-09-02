import ast
from pathlib import Path


def test_deep_agent_uses_deep_context_schema():
    from yuxi.agents.buildin.deep_agent.graph import DeepAgent
    from yuxi.agents.buildin.deep_agent.context import DeepContext

    assert DeepAgent.context_schema is DeepContext


def test_deep_agent_does_not_prepend_context_system_prompt_before_runtime_middleware():
    graph_path = (
        Path(__file__).parents[3] / "package" / "yuxi" / "agents" / "buildin" / "deep_agent" / "graph.py"
    )
    module = ast.parse(graph_path.read_text(encoding="utf-8"))
    create_agent = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "create_agent"
    )
    system_prompt = next(keyword.value for keyword in create_agent.keywords if keyword.arg == "system_prompt")

    assert isinstance(system_prompt, ast.Constant)
    assert system_prompt.value == ""
