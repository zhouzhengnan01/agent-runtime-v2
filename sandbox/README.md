# JetLinks OpenSandbox Skill Images

The runtime calls one adapter path inside every skill image:

```text
/opt/jetlinks/skills/run_skill.py --request /mnt/user-data/workspace/request.json --outputs /mnt/user-data/outputs
```

The request file follows `skill-run.v1`:

```json
{
  "request_schema_version": "skill-run.v1",
  "skill_name": "drawio-generation",
  "thread_id": "thread-id",
  "spec": {},
  "profile": {},
  "workspace_dir": "/mnt/user-data/workspace",
  "outputs_dir": "/mnt/user-data/outputs"
}
```

Build images from the repository root:

```bash
docker build -f sandbox/images/drawio/Dockerfile -t jetlinks/opensandbox-drawio:0.1.0 .
docker build -f sandbox/images/office/Dockerfile -t jetlinks/opensandbox-office:0.1.0 .
docker build -f sandbox/images/code/Dockerfile -t jetlinks/opensandbox-code:0.1.0 .
```

The drawio adapter is implemented now. Office and code images use the same entrypoint so their
adapters can be added behind the same `skill_name` dispatch without changing runtime protocol.

## Skill Dependency Image Cache

Current implementation targets local Docker plus a local OpenSandbox server first. Remote
OpenSandbox is intentionally left as a deployment extension point.

Runtime defaults live in:

```text
config/sandbox/runtime.json
```

Use this file for committed defaults. For machine-specific overrides, create:

```text
config/sandbox/runtime.local.json
```

`runtime.local.json` is ignored by git. Environment variables still have the highest priority and
override both JSON files.

When `SKILL_ENV_CACHE_ENABLED=true`, Python skill dependencies are cached by normalized
`requirements.txt` content instead of by skill name:

```text
skill package requirements.txt
  -> normalize requirements
  -> hash(base image + normalized requirements)
  -> docker image jetlinks-python-skill-deps:<hash-prefix>
```

If a skill package has no `requirements.txt`, no dependency image is created; the sandbox profile
uses its base image directly.

The dependency image contains only the Python dependencies. Skill source code is still mounted or
copied at run time and should not be committed into dependency images. This keeps code changes from
creating new heavy images.

Useful environment variables:

```bash
export SANDBOX_PROVIDER=opensandbox
export SANDBOX_EXECUTOR_ENABLED=true
export SKILL_ENV_CACHE_ENABLED=true
export SKILL_ENV_CACHE_IMAGE_PREFIX=jetlinks-python-skill-deps
export SKILL_ENV_CACHE_MAX_IMAGES=20
export SKILL_ENV_CACHE_MAX_BYTES=21474836480
export SKILL_ENV_CACHE_BUILD_TIMEOUT_SECONDS=1800
```

Torch-heavy tools should share a torch base image instead of installing torch for every skill:

```bash
export SKILL_ENV_CACHE_TORCH_BASE_IMAGE=jetlinks-python-skill-torch-cpu:3.12
```

If `requirements.txt` contains `torch`, `torchvision`, or `torchaudio`, the cache builder uses the
torch base image as the dependency image parent. Build separate torch base images for CPU and CUDA
deployments; do not mix GPU and CPU requirements in one generic profile.

Status and warm-up APIs:

```text
GET  /api/skills/{skill_name}/environment
POST /api/skills/{skill_name}/environment/warm
```

Workbench exposes this through the Skill execution configuration's `预热依赖环境` button.

### TODO

- Local-first: verify OpenSandbox and the API service share the same Docker image namespace. If not,
  the dependency image must be pushed to a registry before OpenSandbox can use it.
- Remote OpenSandbox: add registry push/pull support for dependency images.
- Remote OpenSandbox: add config for image registry, namespace, credentials, and image retention.
- Remote OpenSandbox: optionally move dependency image builds to the OpenSandbox side to avoid
  building images on the API host.
- Add async warm-up jobs so large dependencies such as torch do not block the HTTP request.
- Add UI status polling for `pending`, `warming`, `ready`, `failed`, and `stale` dependency states.
- Add CUDA/GPU profile selection instead of relying only on torch requirement detection.
- Add cache observability: image size, last used time, linked skills, build logs, and failure logs.
- Add an admin cleanup action for stale dependency images.
- Add an allow/block policy for very large dependencies such as torch, opencv, playwright, and
  model downloads.
