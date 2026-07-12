# Environment variables

Parsimony Agents reads only a small set of variables itself. Model providers
and connector packages may read their own credentials.

## Credentials

litellm reads the variable expected by the selected provider:

| Variable | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic models |
| `GEMINI_API_KEY` | Gemini models |
| `OPENAI_API_KEY` | OpenAI models |

Other litellm-supported providers use their documented variables. A provider
key can instead be passed through `Agent(api_key=...)` or `model_config`.

Connector credentials are separate. For example, the FRED connector may read
`FRED_API_KEY`, while explicit code can bind the same value with
`FRED.bind(api_key=...)`. Consult each connector package for its variables.

## Execution

| Variable | Effect |
|---|---|
| `EXECUTOR_CELL_TIMEOUT_S` | Default per-cell timeout for `CodeExecutor`, in seconds. Defaults to `300`. |
| `OCKHAM_DISABLE_SANITIZE` | Values `1`, `true`, or `yes` disable the AST sanitizer. Use for local debugging only. |

Execution mode is selected in code. `Agent()` defaults to the in-process
executor; `create_executor(cwd=..., prefer_boundary=True)` selects bubblewrap on
supported Linux hosts and otherwise falls back in-process with a warning. Pass
`prefer_boundary=False` to request in-process execution explicitly.

There is no `OCKHAM_SANDBOX_BOUNDARY` environment variable. Always inspect the
executor's `capability_tier` before treating it as a security boundary. See
[Code execution](../concepts/code-execution.md).
