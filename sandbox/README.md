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
