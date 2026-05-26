# Secrets Directory

这个目录只用于说明密钥管理约定，不存放真实密钥。

真实密钥来源：

- 模型 API key：推荐通过环境变量注入，例如 `LLM_API_KEY`。
- 加解密主密钥：推荐通过 `JETLINKS_AGENT_SECRET_KEY` 注入。
- 本地开发主密钥：未设置 `JETLINKS_AGENT_SECRET_KEY` 时，运行时会自动生成 `.runtime/secrets/master.key`。

不会提交到 Git 的本地文件：

```text
.env
.runtime/secrets/master.key
config/agents/*.local.json
```

如果需要把明文模型 key 加密成本地 `api_key_enc`：

```bash
uv run python -m app.cli secrets set-api-key --agent default --value "your-model-api-key"
```

完整说明见 [`../../docs/secrets.md`](../../docs/secrets.md)。

