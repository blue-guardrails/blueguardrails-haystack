# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Infer GenAI provider names from Haystack generator component types."""

_PROVIDER_MAP = {
    "OpenAIChatGenerator": "openai",
    "OpenAIGenerator": "openai",
    "OpenAIResponsesChatGenerator": "openai",
    "AzureOpenAIChatGenerator": "azure.ai.openai",
    "AzureOpenAIGenerator": "azure.ai.openai",
    "AzureOpenAIResponsesChatGenerator": "azure.ai.openai",
    "AnthropicChatGenerator": "anthropic",
    "AnthropicGenerator": "anthropic",
    "AnthropicVertexChatGenerator": "gcp.vertex_ai",
    "GoogleAIGeminiChatGenerator": "gcp.gemini",
    "GoogleAIGeminiGenerator": "gcp.gemini",
    "GoogleGenAIChatGenerator": "gcp.gemini",
    "VertexAIGeminiChatGenerator": "gcp.vertex_ai",
    "VertexAIGeminiGenerator": "gcp.vertex_ai",
    "AmazonBedrockChatGenerator": "aws.bedrock",
    "AmazonBedrockGenerator": "aws.bedrock",
}


def infer_provider_name(component_type: str) -> str:
    """Infer the OTel GenAI provider name from a Haystack component type."""
    if component_type in _PROVIDER_MAP:
        return _PROVIDER_MAP[component_type]

    for suffix in ("ChatGenerator", "Generator"):
        if component_type.endswith(suffix):
            provider = component_type[: -len(suffix)].lower()
            return provider if provider else "unknown"
    return "unknown"
