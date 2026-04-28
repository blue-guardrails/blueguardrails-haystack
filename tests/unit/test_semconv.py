# SPDX-FileCopyrightText: 2025-present BlueGuardrails
#
# SPDX-License-Identifier: Apache-2.0

import json

from haystack.dataclasses import ByteStream, ChatMessage, FileContent, ImageContent
from haystack.dataclasses.chat_message import ToolCall
from haystack.tools import Tool

from blueguardrails_haystack.semconv import (
    convert_image_outputs_to_output_messages,
    convert_input_messages,
    convert_output_messages,
    convert_parts_to_input_messages,
    convert_plain_text_to_input_messages,
    convert_plain_text_to_output_messages,
    convert_tool_definitions,
    extract_finish_reason,
    infer_provider_name,
    normalize_finish_reason,
)


class TestConvertInputMessages:
    def test_simple_user_message(self):
        messages = [ChatMessage.from_user("Hello")]
        result = json.loads(convert_input_messages(messages))

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"] == [{"type": "text", "content": "Hello"}]

    def test_system_and_user_messages(self):
        messages = [
            ChatMessage.from_system("You are helpful."),
            ChatMessage.from_user("What is 2+2?"),
        ]
        result = json.loads(convert_input_messages(messages))

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["parts"][0]["content"] == "You are helpful."
        assert result[1]["role"] == "user"

    def test_assistant_with_tool_calls(self):
        tc = ToolCall(tool_name="search", arguments={"query": "weather"}, id="call_1")
        messages = [ChatMessage.from_assistant(text="Let me search.", tool_calls=[tc])]
        result = json.loads(convert_input_messages(messages))

        assert len(result) == 1
        parts = result[0]["parts"]
        assert parts[0] == {"type": "text", "content": "Let me search."}
        assert parts[1] == {"type": "tool_call", "name": "search", "id": "call_1", "arguments": {"query": "weather"}}

    def test_tool_result_message(self):
        origin = ToolCall(tool_name="search", arguments={}, id="call_1")
        messages = [ChatMessage.from_tool(tool_result="Found: sunny", origin=origin)]
        result = json.loads(convert_input_messages(messages))

        assert result[0]["role"] == "tool"
        assert result[0]["parts"][0] == {"type": "tool_call_response", "id": "call_1", "response": "Found: sunny"}

    def test_message_with_name(self):
        messages = [ChatMessage.from_user("Hi", name="alice")]
        result = json.loads(convert_input_messages(messages))
        assert result[0]["name"] == "alice"

    def test_message_without_name(self):
        messages = [ChatMessage.from_user("Hi")]
        result = json.loads(convert_input_messages(messages))
        assert "name" not in result[0]

    def test_multimodal_message_parts(self):
        image = ImageContent(base64_image="aW1hZ2U=", mime_type="image/png", detail="low", validation=False)
        file = FileContent(base64_data="ZmlsZQ==", mime_type="application/pdf", filename="paper.pdf", validation=False)
        messages = [ChatMessage.from_user(content_parts=["Describe these", image, file])]

        result = json.loads(convert_input_messages(messages))
        parts = result[0]["parts"]
        assert parts[0] == {"type": "text", "content": "Describe these"}
        assert parts[1] == {
            "type": "blob",
            "modality": "image",
            "content": "aW1hZ2U=",
            "mime_type": "image/png",
            "detail": "low",
        }
        assert parts[2] == {
            "type": "blob",
            "modality": "document",
            "content": "ZmlsZQ==",
            "mime_type": "application/pdf",
            "filename": "paper.pdf",
        }


class TestConvertOutputMessages:
    def test_simple_reply(self):
        reply = ChatMessage.from_assistant("The answer is 4.", meta={"finish_reason": "stop", "model": "gpt-4o"})
        result = json.loads(convert_output_messages([reply]))

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["parts"] == [{"type": "text", "content": "The answer is 4."}]
        assert result[0]["finish_reason"] == "stop"

    def test_reply_with_tool_calls(self):
        tc = ToolCall(tool_name="calc", arguments={"expr": "2+2"}, id="tc_1")
        reply = ChatMessage.from_assistant(text=None, tool_calls=[tc], meta={"finish_reason": "tool_calls"})
        result = json.loads(convert_output_messages([reply]))

        assert result[0]["finish_reason"] == "tool_call"
        assert result[0]["parts"][0]["type"] == "tool_call"

    def test_missing_finish_reason_is_omitted(self):
        reply = ChatMessage.from_assistant("Hi", meta={})
        result = json.loads(convert_output_messages([reply]))
        assert "finish_reason" not in result[0]

    def test_unknown_finish_reason_is_omitted(self):
        reply = ChatMessage.from_assistant("Hi", meta={"finish_reason": "provider_specific_unknown"})
        result = json.loads(convert_output_messages([reply]))
        assert "finish_reason" not in result[0]

    def test_finish_reason_normalization(self):
        assert normalize_finish_reason("end_turn") == "stop"
        assert normalize_finish_reason("max_tokens") == "length"
        assert normalize_finish_reason("tool_use") == "tool_call"
        assert normalize_finish_reason("refusal") == "content_filter"
        assert normalize_finish_reason("failed") == "error"
        assert normalize_finish_reason("FINISH_REASON_UNSPECIFIED") is None

    def test_extract_finish_reason_from_openai_responses_metadata(self):
        assert extract_finish_reason({"status": "completed"}) == "completed"
        assert extract_finish_reason({"incomplete_details": {"reason": "max_output_tokens"}}) == "max_output_tokens"

        reply = ChatMessage.from_assistant("Hi", meta={"status": "completed"})
        result = json.loads(convert_output_messages([reply]))
        assert result[0]["finish_reason"] == "stop"


class TestPlainTextConversions:
    def test_plain_input(self):
        result = json.loads(convert_plain_text_to_input_messages("Tell me a joke"))
        assert result == [{"role": "user", "parts": [{"type": "text", "content": "Tell me a joke"}]}]

    def test_parts_input(self):
        result = json.loads(convert_parts_to_input_messages(["Describe this input"]))
        assert result == [{"role": "user", "parts": [{"type": "text", "content": "Describe this input"}]}]

    def test_parts_input_with_bytestream(self):
        stream = ByteStream(data=b"hello", mime_type="text/plain")
        result = json.loads(convert_parts_to_input_messages(["Read", stream]))
        assert result[0]["parts"][1] == {
            "type": "blob",
            "modality": "document",
            "content": "aGVsbG8=",
            "mime_type": "text/plain",
        }

    def test_image_output_url(self):
        result = json.loads(convert_image_outputs_to_output_messages(["https://example.com/image.png"]))
        assert result == [
            {
                "role": "assistant",
                "parts": [{"type": "uri", "modality": "image", "uri": "https://example.com/image.png"}],
            }
        ]

    def test_plain_output(self):
        result = json.loads(convert_plain_text_to_output_messages(["Ha ha!", "Another one"]))
        assert len(result) == 2
        assert result[0]["role"] == "assistant"
        assert "finish_reason" not in result[0]
        assert result[0]["parts"][0]["content"] == "Ha ha!"

    def test_plain_output_with_normalized_finish_reasons(self):
        result = json.loads(convert_plain_text_to_output_messages(["Ha ha!"], ["max_tokens"]))
        assert result[0]["finish_reason"] == "length"


class TestToolDefinitions:
    def test_haystack_tool(self):
        tool = Tool(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            function=lambda query: query,
        )
        result = json.loads(convert_tool_definitions([tool]))
        assert result == [
            {
                "type": "function",
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ]

    def test_openai_style_tool_dict(self):
        result = json.loads(
            convert_tool_definitions(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Lookup things",
                            "parameters": {"type": "object"},
                        },
                    }
                ]
            )
        )
        assert result == [
            {"type": "function", "name": "lookup", "description": "Lookup things", "parameters": {"type": "object"}}
        ]


class TestInferProviderName:
    def test_openai_chat(self):
        assert infer_provider_name("OpenAIChatGenerator") == "openai"

    def test_anthropic_chat(self):
        assert infer_provider_name("AnthropicChatGenerator") == "anthropic"

    def test_azure_openai_chat(self):
        assert infer_provider_name("AzureOpenAIChatGenerator") == "azure.ai.openai"

    def test_openai_responses_chat(self):
        assert infer_provider_name("OpenAIResponsesChatGenerator") == "openai"

    def test_azure_openai_responses_chat(self):
        assert infer_provider_name("AzureOpenAIResponsesChatGenerator") == "azure.ai.openai"

    def test_google_genai_chat(self):
        assert infer_provider_name("GoogleGenAIChatGenerator") == "gcp.gemini"

    def test_amazon_bedrock_chat(self):
        assert infer_provider_name("AmazonBedrockChatGenerator") == "aws.bedrock"

    def test_plain_generator(self):
        assert infer_provider_name("OpenAIGenerator") == "openai"

    def test_unknown(self):
        assert infer_provider_name("SomethingElse") == "unknown"

    def test_bare_generator(self):
        assert infer_provider_name("Generator") == "unknown"
