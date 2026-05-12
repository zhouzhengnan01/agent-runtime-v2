# Secrets and Encrypted Model Keys

本项目支持两种模型密钥来源：

- 推荐方式：通过环境变量注入，例如 `LLM_API_KEY`。
- 本地兼容方式：把 agent local override 里的 `model.api_key` 自动加密成 `model.api_key_enc`。

不要把真实 API key、`JETLINKS_AGENT_SECRET_KEY` 或 `.runtime/secrets/master.key` 提交到 GitHub。

## 目录约定

```text
config/secrets/README.md          # Git 中可见的密钥目录说明，不放真实密钥
.runtime/secrets/master.key       # 本地自动生成的加解密主密钥，不提交
config/agents/*.local.json        # 本地 agent 覆盖配置，不提交
```

`.runtime/`、`.env`、`config/agents/*.local.json` 都已经在 `.gitignore` 中忽略。GitHub 上看不到
`.runtime/secrets/master.key` 是预期行为，因为它是真实主密钥。

## 推荐：环境变量注入

应用模板里的模型配置通常使用：

```json
{
  "model": {
    "api_key_env": "LLM_API_KEY"
  }
}
```

运行前设置：

```bash
export LLM_API_KEY="your-model-api-key"
```

Java 或进程管理器启动 Runtime 时，也可以把 `LLM_API_KEY` 放入子进程环境变量。这个方式最适合生产部署，
因为仓库和配置文件里都不会出现密钥明文或密文。

## 本地加密：api_key_enc

如果需要在本地 agent 覆盖配置中保存模型 key，使用命令写入加密值：

```bash
uv run python -m app.cli secrets set-api-key --agent default --value "your-model-api-key"
```

等价的虚拟环境命令：

```bash
.venv/bin/python -m app.cli secrets set-api-key --agent default --value "your-model-api-key"
```

命令会写入：

```text
config/agents/default.local.json
.runtime/secrets/master.key
```

`config/agents/default.local.json` 中保存的是密文：

```json
{
  "model": {
    "api_key_enc": "enc.fernet.v1...."
  }
}
```

运行时加载 agent 配置时，会自动把 `model.api_key_enc` 解密成内存里的 `model.api_key`，再调用
OpenAI-compatible 模型服务。详情接口和 `python -m app.cli show-agent` 会把 `model.api_key` 和
`model.api_key_enc` 脱敏成 `********`。

## 应用模板模型密钥

如果密钥属于应用模板模型配置，也就是 `config/apps/*.json` 的 `models[].api_key_enc`，需要使用 app 专用
purpose 加密。不要使用 `secrets set-api-key --agent ...`，那个命令只适用于
`config/agents/*.local.json` 的 `model.api_key_enc`。

应用模板模型密钥使用：

```bash
uv run python -m app.cli secrets encrypt-app-api-key \
  --app "_debug-1c81a222b0fa9000" \
  --model "gpt-5.5" \
  --value "your-model-api-key"
```

命令会把密文打印到 stdout，可放入：

```json
{
  "models": [
    {
      "name": "gpt-5.5",
      "model": "gpt-5.5",
      "api_key_enc": "enc.fernet.v1...."
    }
  ]
}
```

运行时解密 `models[].api_key_enc` 时会优先使用：

```text
app:<app-name>:model:<model-name>:api_key
```

其中 `<model-name>` 对应 app JSON 中选中模型的 `name`，如果没有 `name`，再使用 `model` 或
`default_model`。例如：

```text
app:_debug-1c81a222b0fa9000:model:gpt-5.5:api_key
```

agent local override 的 purpose 是：

```text
agent:default:model:api_key
```

这两类密文不能互换。报错 `encrypted secret authentication failed` 通常表示主密钥不一致；报错
`encrypted secret purpose mismatch` 才表示主密钥一致但 purpose 不一致。

## 加解密主密钥

`api_key_enc` 不是独立可恢复的密文，必须配合加密时使用的主密钥才能解密。主密钥来源优先级：

```text
JETLINKS_AGENT_SECRET_KEY -> .runtime/secrets/master.key
```

固定部署环境推荐显式设置：

```bash
export JETLINKS_AGENT_SECRET_KEY="replace-with-a-long-random-secret"
uv run python -m app.cli secrets set-api-key --agent default --value "your-model-api-key"
```

`JETLINKS_AGENT_SECRET_KEY` 可以是 Fernet key，也可以是普通长随机字符串。普通字符串会通过 SHA-256
派生成 Fernet 可用密钥；只要字符串不变，历史 `api_key_enc` 就能继续解密。

如果没有设置 `JETLINKS_AGENT_SECRET_KEY`，首次加密或首次加载明文 `model.api_key` 时会自动生成：

```text
.runtime/secrets/master.key
```

这个文件只适合单机本地开发。迁移到新机器时，如果要沿用旧的 `api_key_enc`，必须同时迁移
`.runtime/secrets/master.key`，或者在新环境配置相同的 `JETLINKS_AGENT_SECRET_KEY`。主密钥丢失或更换后，
旧的 `api_key_enc` 无法解密，需要重新执行 `secrets set-api-key`。

## 明文迁移

历史 local config 如果临时写了明文：

```json
{
  "model": {
    "api_key": "your-model-api-key"
  }
}
```

运行时第一次加载该 agent 时会自动改写为：

```json
{
  "model": {
    "api_key_enc": "enc.fernet.v1...."
  }
}
```

本次加载仍会在内存中使用原始 key；后续加载会从 `api_key_enc` 自动解密。

## 安全边界

- 可以提交：`config/apps/*.json` 中的 `api_key_env`、`config/secrets/README.md`、文档说明。
- 不要提交：真实 API key、`.env`、`.runtime/secrets/master.key`、`JETLINKS_AGENT_SECRET_KEY`。
- 谨慎提交：`api_key_enc`。只有在主密钥绝不泄漏的前提下才可以提交密文；密文和主密钥同时泄漏等同于明文泄漏。
- 依赖要求：启用 `api_key_enc` 时，运行环境必须安装 `cryptography`。使用 `uv sync`、`.venv` 或项目启动脚本即可。
