from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_core.messages import AIMessage, AIMessageChunk

from yuxi.agents import BaseAgent, load_chat_model

from .context import (
    MODEL_CONTEXT_WINDOW_TOKENS,
    MedicationReviewAcmBoundedContext,
)
from .context_view import AcmBoundedModelViewMiddleware
from .controller import AcmBoundedControllerMiddleware
from .generation import AcmGenerationGuardMiddleware
from .harness import AcmBoundedHarnessMiddleware, METHOD_VERSION
from .models import ActionDirective, MedicationReviewAcmBoundedState
from .tools import BOUNDED_TOOLS


def _declare_context_window(
    model: Any,
    context: MedicationReviewAcmBoundedContext,
) -> Any:
    profile = dict(getattr(model, "profile", None) or {})
    provider_limit = profile.get("max_input_tokens")
    verified = isinstance(provider_limit, int) and provider_limit > 0
    effective_limit = min(provider_limit, MODEL_CONTEXT_WINDOW_TOKENS) if verified else MODEL_CONTEXT_WINDOW_TOKENS
    profile["max_input_tokens"] = effective_limit
    setattr(context, "_acm_bounded_context_window_verified", verified)
    setattr(
        context,
        "_acm_bounded_provider_context_window_tokens",
        effective_limit,
    )
    try:
        model.profile = profile
        return model
    except (AttributeError, TypeError, ValueError):
        model_copy = getattr(model, "model_copy", None)
        if callable(model_copy):
            return model_copy(update={"profile": profile})
        raise RuntimeError("当前模型对象不支持声明 max_input_tokens") from None


class MedicationReviewAcmBoundedAgent(BaseAgent):
    name = "有界控制式用药调查智能体（ACM Bounded）"
    description = (
        "独立于现有 RNMI 的 adaptive coverage 实现：以确定性控制器逐义务调度，"
        "使用窄工具、按相关性重建模型视图、保留 Evidence 原文，并在实际 provider "
        "上下文容量（最高 262144 token）内运行；不设置阶段输入硬线或固定调查/检索 K。"
    )
    capabilities: list[str] = []
    context_schema = MedicationReviewAcmBoundedContext
    metadata = {
        "examples": ["审查该患者完整治疗方案的合理性，并给出逐项判断、修正建议和依据。"],
        "method_family": "acm-prim-rag-v9",
        "method_version": METHOD_VERSION,
        "trace_schema_version": "13.0",
        "atlas_schema_version": "3.0",
        "model_context_window_tokens": MODEL_CONTEXT_WINDOW_TOKENS,
    }

    def project_stream_message(
        self,
        message: Any,
        state_values: dict[str, Any],
        *,
        context: MedicationReviewAcmBoundedContext | None = None,
    ) -> Any:
        """Hide model prose from action rounds while preserving tool-call chunks."""
        if not isinstance(message, (AIMessage, AIMessageChunk)):
            return message
        raw_directive = state_values.get("action_directive") or getattr(
            context,
            "_acm_bounded_directive",
            None,
        )
        if raw_directive is None:
            return message
        directive = (
            raw_directive
            if isinstance(raw_directive, ActionDirective)
            else ActionDirective.model_validate(raw_directive)
        )
        if (
            directive.expected_output_kind != "tool_call"
            or message.additional_kwargs.get("acm_bounded_terminal_failure") is True
        ):
            return message
        additional_kwargs = dict(message.additional_kwargs)
        for key in ("reasoning_content", "reasoning", "analysis"):
            additional_kwargs.pop(key, None)
        return message.model_copy(update={"content": "", "additional_kwargs": additional_kwargs})

    async def get_graph(
        self,
        context: MedicationReviewAcmBoundedContext | None = None,
        **kwargs: Any,
    ):
        del kwargs
        context = context or self.context_schema()
        model = _declare_context_window(load_chat_model(context.model), context)
        return create_agent(
            model=model,
            tools=BOUNDED_TOOLS,
            system_prompt="",
            middleware=[
                AcmBoundedHarnessMiddleware(model=model),
                AcmBoundedControllerMiddleware(),
                AcmBoundedModelViewMiddleware(),
                AcmGenerationGuardMiddleware(),
                ModelRetryMiddleware(
                    max_retries=context.technical_retry_limit,
                    on_failure="error",
                ),
            ],
            state_schema=MedicationReviewAcmBoundedState,
            context_schema=MedicationReviewAcmBoundedContext,
            checkpointer=await self._get_checkpointer(),
            name="medication_review_acm_bounded",
        )
