# blueguardrails-haystack

BlueGuardrails tracing integration for Haystack.

The package sends Haystack generator calls to BlueGuardrails as OpenTelemetry GenAI spans. It records inputs, outputs, model metadata, request options, tool definitions, token usage, and finish reasons for LLM calls in Haystack pipelines and agents.

The connector installs as a sidecar. Your existing Haystack tracer keeps working, and BlueGuardrails receives only generator spans.

## Install

Install the package with pip:

```bash
pip install blueguardrails-haystack
```

Or with uv:

```bash
uv add blueguardrails-haystack
```

The package supports Python 3.11 through 3.14.

Set your BlueGuardrails API key:

```bash
export BLUEGUARDRAILS_API_KEY="your-api-key"
```

Set your model provider credentials as usual. For example, set `OPENAI_API_KEY` when you use `OpenAIChatGenerator`.

Install any provider-specific Haystack integrations you use. For example, install `anthropic-haystack`, `google-genai-haystack`, or `amazon-bedrock-haystack` if your pipeline uses those generators.

## Use

Add `BlueGuardrailsConnector` to your pipeline. You don't need to connect it to other components.

```python
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from blueguardrails_haystack import BlueGuardrailsConnector

pipe = Pipeline()

pipe.add_component(
    "blueguardrails",
    BlueGuardrailsConnector(
        name="support-bot",
        api_key=Secret.from_env_var("BLUEGUARDRAILS_API_KEY"),
        tags={"environment": "development"},
    ),
)

pipe.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))

result = pipe.run(
    {
        "llm": {
            "messages": [
                ChatMessage.from_user("Reply in one sentence. What is Haystack?")
            ]
        }
    }
)

print(result["llm"]["replies"][0].text)
```

When the pipeline runs, BlueGuardrails receives a trace for the `llm` generator call.

### Trace an agent

To trace a standalone Haystack agent, configure the BlueGuardrails tracer before you run the agent.

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from blueguardrails_haystack import configure_blueguardrails_tracer


def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."


configure_blueguardrails_tracer(
    name="support-agent",
    tags={"environment": "development"},
)

weather_tool = Tool(
    name="get_weather",
    description="Returns the weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    function=get_weather,
)

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    tools=[weather_tool],
    system_prompt="Use tools when they help answer the user.",
    exit_conditions=["text"],
    max_agent_steps=3,
)

result = agent.run(
    messages=[ChatMessage.from_user("What is the weather in Berlin?")]
)

print(result["last_message"].text)
```

BlueGuardrails receives a trace for each generator call the agent makes.

`configure_blueguardrails_tracer()` reads `BLUEGUARDRAILS_API_KEY` by default. If you don't set `BLUEGUARDRAILS_API_KEY`, pass `api_key` explicitly. It raises `ValueError` if neither is set:

```python
configure_blueguardrails_tracer(name="support-agent", api_key="your-api-key")
```

## Configure the connector

```python
BlueGuardrailsConnector(
    name="production-rag",
    api_key=Secret.from_env_var("BLUEGUARDRAILS_API_KEY"),
    sample_rate=0.1,
    tags={"environment": "production", "team": "search"},
)
```

| Argument | Default | Description |
| --- | --- | --- |
| `name` | Required | Trace name shown in BlueGuardrails. |
| `api_key` | `Secret.from_env_var("BLUEGUARDRAILS_API_KEY")` | API key for trace export. |
| `endpoint` | `https://api.blueguardrails.com/v1/traces` | OpenTelemetry Protocol (OTLP) HTTP traces endpoint. |
| `sample_rate` | `1.0` | Fraction of generator calls to export. Use a value between `0.0` and `1.0`. |
| `tags` | `None` | Conversation tags attached to each exported generator span. |

Traces include generator inputs and outputs. Review your data handling requirements before you enable the connector in production.

## License

Apache-2.0
