# Eredu for Mesh

Run a local MLX model on your Apple Silicon Mac with
[Eredu](https://github.com/jbg/eredu), served through Mesh's normal
OpenAI-compatible API.

**Early access:** Apple Silicon (macOS arm64) only. Text chat and streaming;
tool calls are rejected for now.

## 1. Install

```sh
mesh-llm plugins install Mesh-LLM/eredu-plugin
```

This downloads the latest release built from `main`. It is a single binary
(ad-hoc signed, not notarized). It never downloads model weights.

## 2. Pick a model

Point it at a local MLX safetensors model directory (with `config.json`,
`tokenizer.json` and `*.safetensors`). For example, if you already have
`mlx-community/Qwen3-1.7B-4bit` in your Hugging Face cache:

```sh
ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen3-1.7B-4bit/snapshots/*
```

## 3. Add it to Mesh

Add to `~/.mesh-llm/config.toml` (edit an existing `[runtime]` section rather
than adding a second one):

```toml
[runtime]
mode = "on_demand"

[[plugin]]
name = "eredu-plugin"
args = ["--model", "/absolute/path/to/model-dir", "--model-id", "qwen3-1.7b-eredu"]
```

Then:

```sh
mesh-llm serve
curl http://127.0.0.1:9337/v1/models
curl http://127.0.0.1:9337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-1.7b-eredu","messages":[{"role":"user","content":"Hello"}]}'
```

The model loads when Mesh starts the plugin and appears in the model list
once loaded. Stopping Mesh stops the plugin.

## Try it without Mesh

```sh
~/.mesh-llm/plugins/installed/eredu-plugin/eredu-plugin --model /path/to/model-dir --port 8080
# then talk to http://127.0.0.1:8080/v1
```

## Development

`just verify` (mock tests, fmt, Clippy), `just build` (native MLX build; needs
Xcode with the Metal toolchain: `xcodebuild -downloadComponent MetalToolchain`),
`just package v0.1.0` (release archive in `dist/`), `just clean`.

Every push to `main` publishes a release (`v0.1.<run>`), so
`mesh-llm plugins install Mesh-LLM/eredu-plugin` always gets the latest.

Eredu is MIT OR Apache-2.0; this plugin is Apache-2.0. Model weights have
their own licences.
