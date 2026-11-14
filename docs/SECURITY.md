# Security & secrets policy (public repository)

## Hard rules

1. **Never commit secrets.** No `.env`, tokens, API keys, private URLs,
   personal paths. Config comes from environment variables; `.env.example`
   (values sanitized) is the only env file in the repo.
2. `.gitignore` blocks `.env*` and key/credential patterns — do not weaken it.
3. CI runs **gitleaks** on every push and PR. A red CI means fix before merge.
4. If a secret still lands in history: rotate/revoke the secret **first**,
   then clean history (`git filter-repo`), then force-push. Rotation is the
   fix; history rewriting alone is not.

## System-level posture

- The brain's WebSocket is published on **`127.0.0.1` only** — the gesture
  service is unreachable from LAN.
- **Double whitelist**: a command executes on the host only if its action is
  in the brain's registry AND every key name resolves in the edge's compiled
  key table. Neither YAML nor wire payloads can inject arbitrary keystrokes.
- `pause_system` is meta: it never reaches the keyboard layer.
- MediaPipe model is downloaded from Google's pinned URL at image build time;
  no model binaries live in git.

## Trust model

The edge agent trusts only the brain it connects to, on loopback. That is
acceptable because both halves ship from this repo and run on one machine.
Do not expose the WS port; do not run the brain on a remote host without
adding authentication first.
