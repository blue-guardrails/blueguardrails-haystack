<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/logo-wordmark-white-transparent.svg">
  <img alt="Blue Guardrails" src="./assets/logo-wordmark-blue-transparent.svg">
</picture>

# Blue Guardrails - Haystack

The Blue Guardrails Haystack integration instruments LLM and agent calls in Haystack and sends them as OpenTelemetry traces to the Blue Guardrails platform.

Use Blue Guardrails to monitor your agents and other GenAI applications in production or evaluate pre-deployment.

The Blue Guardrails reliability layer runs on ingested traces and detects issues like hallucinations, poor instruction-following, or tool-calling issues.

More info on Blue Guardrails in the official [documentation](https://docs.blueguardrails.com).

## Features

| Feature | Description |
| --- | --- |
| Haystack tracing | Trace Haystack agents and direct LLM calls. |
| OpenTelemetry GenAI compatibility | Export traces that follow the OpenTelemetry [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). |
| Rich trace data | Capture inputs, outputs, tool calls, tool definitions, model parameters, and usage. |
| Sampling controls | Choose how much trace data to send with configurable sampling. |
| Reliability layer | Use the Blue Guardrails reliability layer to help prevent your agent from going off track. |

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

## Use

### Trace an agent

To trace a Haystack agent, configure the Blue Guardrails tracer before you run the agent.

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

Blue Guardrails receives a trace for each generator call the agent makes.

`configure_blueguardrails_tracer()` reads `BLUE_GUARDRAILS_API_KEY` by default. If you don't set `BLUE_GUARDRAILS_API_KEY`, pass `api_key` explicitly. It raises `ValueError` if neither is set:

```python
configure_blueguardrails_tracer(name="support-agent", api_key="your-api-key")
```

### Trace a pipeline

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
        api_key=Secret.from_env_var("BLUE_GUARDRAILS_API_KEY"),
        tags={"environment": "development"},
    ),
)

pipe.add_component("llm", OpenAIChatGenerator(model="gpt-5.4-mini"))

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

When the pipeline runs, Blue Guardrails receives a trace for the `llm` call.

### Load and trace a pipeline from YAML

Haystack pipelines can be serialized to and loaded from YAML. Put the connector in the YAML just like any other component; loading the pipeline initializes Blue Guardrails tracing.

`pipeline.yaml`:

<!-- blueguardrails-yaml-example:start -->
```yaml
components:
  blueguardrails:
    type: blueguardrails_haystack.components.connector.BlueGuardrailsConnector
    init_parameters:
      name: support-bot
      api_key:
        type: env_var
        env_vars:
          - BLUE_GUARDRAILS_API_KEY
        strict: true
      sample_rate: 1.0
      tags:
        environment: development
  llm:
    type: haystack.components.generators.chat.openai.OpenAIChatGenerator
    init_parameters:
      model: gpt-5.4-mini
      api_key:
        type: env_var
        env_vars:
          - OPENAI_API_KEY
        strict: true
connections: []
```
<!-- blueguardrails-yaml-example:end -->

Set `BLUE_GUARDRAILS_API_KEY` and `OPENAI_API_KEY` in your environment, then load and run it:

```python
from pathlib import Path

from haystack import Pipeline
from haystack.dataclasses import ChatMessage

pipeline = Pipeline.loads(Path("pipeline.yaml").read_text())

result = pipeline.run(
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

Use `Pipeline.dumps()` or `Pipeline.dump()` to serialize a Python-built pipeline back to YAML.

## Configure the connector

```python
BlueGuardrailsConnector(
    name="production-rag",
    api_key=Secret.from_env_var("BLUE_GUARDRAILS_API_KEY"),
    sample_rate=0.1,
    tags={"environment": "production", "team": "search"},
)
```

| Argument | Default | Description |
| --- | --- | --- |
| `name` | Required | Trace name shown in Blue Guardrails. |
| `api_key` | `Secret.from_env_var("BLUE_GUARDRAILS_API_KEY")` | API key for trace export. |
| `endpoint` | `https://api.blueguardrails.com/v1/traces` | OpenTelemetry Protocol (OTLP) HTTP traces endpoint. |
| `sample_rate` | `1.0` | Fraction of generator calls to export. Use a value between `0.0` and `1.0`. |
| `tags` | `None` | Conversation tags attached to each exported generator span. |

Traces include generator inputs and outputs. Review your data handling requirements before you enable the connector in production.

## License

Apache-2.0
