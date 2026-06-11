#!/usr/bin/env sh
set -eu

AUTHORIZED_KEYS_FILE="${AUTHORIZED_KEYS_FILE:-/etc/debug-frp/authorized_keys}"
FRPC_CONFIG_FILE="${FRPC_CONFIG_FILE:-/etc/frp/frpc.ini}"

if [ ! -s "$AUTHORIZED_KEYS_FILE" ]; then
  echo "missing authorized_keys file: $AUTHORIZED_KEYS_FILE" >&2
  exit 1
fi

if [ ! -s "$FRPC_CONFIG_FILE" ]; then
  echo "missing frpc config file: $FRPC_CONFIG_FILE" >&2
  exit 1
fi

mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
cp "$AUTHORIZED_KEYS_FILE" /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

ssh-keygen -A >/dev/null 2>&1 || true

cat >/etc/ssh/sshd_config.d/99-ai-agent-debug.conf <<'EOF'
PermitRootLogin prohibit-password
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
AllowTcpForwarding yes
X11Forwarding no
EOF

/usr/sbin/sshd -D -e &
sshd_pid="$!"

/usr/local/bin/frpc -c "$FRPC_CONFIG_FILE" &
frpc_pid="$!"

term_handler() {
  kill "$sshd_pid" "$frpc_pid" 2>/dev/null || true
  wait "$sshd_pid" "$frpc_pid" 2>/dev/null || true
}
trap term_handler INT TERM

wait -n "$sshd_pid" "$frpc_pid"
term_handler
