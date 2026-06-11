# ai-agent debug FRP

This directory defines FRP/SSH debugging for inspecting the ACK runtime image through FRP on `110.40.237.78`.

The production `stable-amd64` image includes `sshd` and `frpc`, but it does not bake SSH keys or FRP tokens into the image. The debug tunnel only starts when Kubernetes mounts `/etc/debug-frp/authorized_keys` and either `/etc/frp/frpc.ini` or the FRP environment variables are provided.

## Prebuilt debug image

For the fastest workflow, build a separate debug image that already includes:

- `/etc/debug-frp/authorized_keys`
- `JETLINKS_AGENT_DEBUG_FRP_SERVER_ADDR`
- `JETLINKS_AGENT_DEBUG_FRP_SERVER_PORT`
- `JETLINKS_AGENT_DEBUG_FRP_TOKEN`

This image still defaults to `JETLINKS_AGENT_DEBUG_FRP_ENABLED=0`, so runtime only needs:

```yaml
- name: JETLINKS_AGENT_DEBUG_FRP_ENABLED
  value: "1"
- name: JETLINKS_AGENT_DEBUG_FRP_REMOTE_PORT
  value: "30089"
```

Then connect with the matching private key:

```bash
ssh -i /Users/chenhao/Desktop/code/server-manager/.tmp/ai-agent-debug-frp \
  -p 30089 \
  root@110.40.237.78
```

Build and push:

```bash
docker build \
  -f deploy/debug-frp/Dockerfile.debug-frp-amd64 \
  -t registry.cn-hangzhou.aliyuncs.com/koudaimao/jetlinks-agent-runtime-v2:debug-frp-amd64 \
  .

docker push registry.cn-hangzhou.aliyuncs.com/koudaimao/jetlinks-agent-runtime-v2:debug-frp-amd64
```

## Relay

- FRP server: `110.40.237.78:443`
- Debug remote SSH port: `30088`
- Kubernetes namespace: `vanke`
- Production PVC mounted read/write in the same paths as the `ai-agent` workload.

## Production Image Built-in Debug

To enable debug access on the production Pod, mount a Secret with these files:

```text
/etc/debug-frp/authorized_keys
/etc/frp/frpc.ini
```

The container entrypoint starts `sshd` and `frpc` before launching uvicorn. If the files are missing, it logs a skip message and continues normal service startup.

## Create a Temporary SSH Key

```bash
mkdir -p /Users/chenhao/Desktop/code/server-manager/.tmp
ssh-keygen -t ed25519 \
  -f /Users/chenhao/Desktop/code/server-manager/.tmp/ai-agent-debug-frp \
  -N '' \
  -C ai-agent-debug-frp
```

Put the `.pub` content into a local Secret manifest under `authorized_keys`.

## Secret

Copy `secret.yaml.example` to a local untracked file, replace `REPLACE_WITH_FRP_TOKEN_FROM_110`, then apply it:

```bash
kubectl apply -f deploy/debug-frp/secret.local.yaml
```

Mount it into the production workload:

```yaml
volumeMounts:
  - mountPath: /etc/debug-frp
    name: ai-agent-debug-frp
    readOnly: true
  - mountPath: /etc/frp
    name: ai-agent-debug-frp
    readOnly: true
volumes:
  - name: ai-agent-debug-frp
    secret:
      secretName: ai-agent-debug-frp
      defaultMode: 0400
```

After the Pod starts and FRP registers the remote port, connect through 110:

```bash
ssh -i /Users/chenhao/Desktop/code/server-manager/.tmp/ai-agent-debug-frp \
  -p 30088 \
  root@110.40.237.78
```

## Disable

Set `JETLINKS_AGENT_DEBUG_FRP_ENABLED=0`, or remove the Secret mount and restart the Pod.

## Temporary Debug Pod

For a separate debug Pod instead of the production Pod, build and push:

```bash
docker build \
  -f deploy/debug-frp/Dockerfile.debug-frp \
  -t registry.cn-hangzhou.aliyuncs.com/koudaimao/jetlinks-agent-runtime-v2:debug-frp \
  .

docker push registry.cn-hangzhou.aliyuncs.com/koudaimao/jetlinks-agent-runtime-v2:debug-frp
```

Then deploy:

```bash
kubectl apply -f deploy/debug-frp/secret.local.yaml
kubectl apply -f deploy/debug-frp/pod.yaml
kubectl -n vanke logs -f pod/ai-agent-debug-frp
```

## Remove

```bash
kubectl -n vanke delete pod ai-agent-debug-frp
kubectl -n vanke delete secret ai-agent-debug-frp
rm -f /Users/chenhao/Desktop/code/server-manager/.tmp/ai-agent-debug-frp*
```
