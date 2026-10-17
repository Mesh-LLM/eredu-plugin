# Eredu Mesh plugin

Standalone application using pinned public Eredu APIs and Mesh plugin SDK; no engine or Mesh host fork.
- Work on feature branches. No main pushes, merges or releases without Mic's explicit instruction.
- Keep inference on a loopback HTTP data plane; IPC handles control and lifecycle only.
- One explicit local model; never download weights, mutate Mesh configuration or stop unrelated processes.
- Apple Silicon first. Default tests use mocks, never load real models or touch credentials.
- Native model objects belong to a dedicated owning worker; bound admission and output buffering, cancel on disconnect and shutdown.
- Advertise readiness only after loading. Advertise tool support only with validated native semantics.
- Use just build, just verify and just clean; Cargo serially. Run full package tests and warning-denying Clippy before commits.
- Read Mesh manage-ci skill before any CI work. Use configured git identity/signing without overrides.
- Preserve model files and live-agent worktrees; clean only owned build artifacts after validation.
