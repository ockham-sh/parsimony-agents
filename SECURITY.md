# Security Policy

## Supported Versions

| Package | Version | Supported |
|---------|---------|-----------|
| parsimony | 0.1.x | Yes |
| parsimony-agents | 0.1.x | Yes |

## Reporting a Vulnerability

**Please do not open public GitHub issues for security vulnerabilities.**

Instead, email **security@ockham.sh** with:

- A description of the vulnerability
- Steps to reproduce (if applicable)
- The affected package(s) and version(s)
- Any potential impact you've identified

### What to expect

- **48 hours**: We will acknowledge your report
- **7 days**: We will provide an initial assessment and estimated timeline
- **30 days**: We aim to release a fix for confirmed vulnerabilities

We will coordinate with you on disclosure timing. We follow responsible disclosure practices and will credit reporters in release notes (unless you prefer to remain anonymous).

## Scope

This policy covers the open-source packages:
- `parsimony` (parsimony-core)
- `parsimony-agents`
- `parsimony-connectors` (parsimony-*)

For vulnerabilities in the ockham terminal (coming soon) (the AGPLv3 agentic data-analysis product built on this library), please also email security@ockham.sh.

## Code execution and credential isolation

Agent-authored code is untrusted. The security model rests on two boundaries, not
on inspecting the code:

- **The kernel holds no credentials.** A connector is exposed to agent code as a
  `RemoteConnector` — a name-routed stub carrying only the connector's name and
  the supervisor socket, with no `fn`, `bound_arguments`, `secrets`, or even
  metadata. The bound, credentialed connector lives in a trusted supervisor
  process; the kernel invokes it only by RPC to a broker there, which means a
  bound connector is the sole egress path.
- **The kernel is confined.** `parsimony_agents.execution.sandbox` runs the kernel
  out-of-process. On Linux with unprivileged user namespaces, the kernel is
  spawned under `bwrap` (`confine=True`), which removes network access, clears the
  environment, and confines the filesystem to the workspace
  (`capability_tier == "namespaces"`). Where no boundary is
  available (non-Linux self-host), execution falls back in-process
  (`capability_tier == "none"`) and says so loudly.

The compile-time guard (`sanitize.assert_safe_code`) and any host-side env
scrubbing are **best-effort defense-in-depth for the in-process fallback only**;
they are trivially bypassable and must never be treated as containment. The
boundary is the out-of-process kernel. Hosts that run untrusted code should verify
`capability_tier` (surfaced on the executor's health endpoint) rather than assume
a boundary is present.
