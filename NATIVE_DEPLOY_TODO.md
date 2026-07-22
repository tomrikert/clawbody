# Reachy Native Deployment TODO

Last checked: 2026-07-22

Goal: package and deploy ClawBody as a native Reachy Mini app that starts and recovers without the laptop, while continuing to use the existing OpenClaw `main` agent and `agent:main:reachy` session. Do not create a second Asmo agent.

## Work queue

- [x] **Consolidate reviewer findings and inspect the current native app contract**
  - Capture the actionable Claude review findings in the repo.
  - Verify the expected `ReachyMiniApp` class, lifecycle hooks, metadata, and runtime assumptions against the installed Reachy 1.9 tooling.
  - Exit condition: findings are recorded and the exact contract changes are identified.
  - Evidence: `NATIVE_APP_REVIEW.md`; installed Reachy 1.9 validator reproduces the entry-point/class failure and documents the required lifecycle corrections.

- [x] **Correct the native Reachy app entry point and lifecycle**
  - Fix the app class/entry-point contract.
  - Ensure startup, stop, cleanup, and robot-resource ownership are safe and idempotent.
  - Preserve the existing xeon STT/TTS and `agent:main:reachy` architecture.
  - Exit condition: native entry point imports and initializes under the official tooling.
  - Evidence: `ReachyMiniOpenclaw(ReachyMiniApp)` is the package entry point; external stop events cancel all app tasks, shutdown is async/idempotent, native failures propagate, and native camera tracking uses localhost.

- [x] **Add reconnect and watchdog resilience**
  - Recover from Reachy daemon, WebRTC, xeon voice service, and OpenClaw gateway interruptions without a laptop.
  - Use bounded backoff and clean shutdown; do not create restart storms.
  - Exit condition: automated tests cover disconnect/reconnect and process restart behavior.
  - Evidence: gateway reconnect, transient STT/TTS retry, microphone/speaker recovery backoff, and native failure propagation to the Reachy supervisor are covered by the 33-test suite.

- [x] **Make native configuration and secrets deployment-safe**
  - Remove LAN-specific or workstation-only assumptions from the package.
  - Keep secrets out of git and Hugging Face history.
  - Document the minimum runtime configuration for Reachy, xeon STT/TTS, and the existing OpenClaw gateway/session.
  - Exit condition: a clean install can be configured without editing source.
  - Evidence: service endpoints are explicit required configuration, deployment environment overrides local `.env`, native configuration is documented in `README.md`, and secrets remain environment-only.

- [x] **Run local validation and final code review**
  - Run the complete test suite.
  - Run `reachy-mini-app-assistant check` (or the current official equivalent).
  - Review the final diff for safety, secret leakage, lifecycle correctness, and accidental creation of an Asmo(deux).
  - Exit condition: validation passes with no unresolved critical findings.
  - Evidence: UV-managed CPython 3.13.12; 35 tests pass; the Reachy 1.9 app assistant passes its clean install, entry-point, and uninstall checks; package builds succeed; artifact and workspace scans found no credential material; final review fixed executor-thread leakage, un-awaited CLI/Gradio shutdown, and blocking async startup sleeps.

- [x] **Package and publish a private Hugging Face Space**
  - Use the Hugging Face credentials in 1Password vault `Asmo`, item `3nm3rm35entwmzoc64bznvdbma` at runtime only.
  - Create/update the private native Reachy app Space and push the validated package.
  - Never print or persist the token in logs, repo files, shell history, or memory.
  - Exit condition: the private Space exists and exposes the expected app revision.
  - Evidence: the scoped write token was read from 1Password at runtime only; the official Reachy 1.9 publish check passed; private Space `asmolebot/clawbodylocal` was created and verified at revision `dc3d1ae0c2d4386ed186476edc6efb388d08c97d` with the expected native app files and no secret-like filenames. The original `asmolebot/clawbody` identifier resolves to the renamed Space through the authenticated Hub API.

- [ ] **Install and configure the app on Reachy Mini**
  - HOLD: do not begin this item until Chris explicitly signs off on the final deployment.
  - Install through the Reachy app API/tooling.
  - Configure it as the startup app if supported by the installed Reachy version.
  - Keep deployment reversible and record the prior app/startup state.
  - Exit condition: the app launches on Reachy without the laptop console.

- [ ] **Run laptop-independent end-to-end acceptance tests**
  - Verify microphone -> xeon STT -> existing OpenClaw Asmo session -> xeon TTS -> speaker.
  - Verify camera/face/sound tracking, representative emotions, and at least one safe dance.
  - Restart the app/daemon and confirm recovery.
  - Exit condition: a complete spoken turn and motion test pass while the laptop is closed/disconnected.

- [ ] **Commit, push, and update the upstream draft PR**
  - Commit only reviewed source/docs/config changes; exclude private deployment state and credentials.
  - Push the existing branch and update the draft PR with native-deployment validation results.
  - Exit condition: clean worktree, pushed commit, and PR reflects the deployed revision.

- [ ] **Begin the separate body-latency/bootstrap optimization**
  - Only after native deployment is stable, slim the Reachy session bootstrap/tool surface and add first-token/first-audio timing.
  - Do not create a second agent personality or duplicate long-term memory.
  - Exit condition: a measured optimization plan is recorded from the native deployment baseline.

## Heartbeat rules

- Advance at most one unchecked item per heartbeat.
- Read current repo/runtime state before acting; do not redo completed work.
- Mark an item complete only when its exit condition is evidenced.
- Continue safe local implementation and validation autonomously.
- Publishing/staging the private Space is authorized. Installing, launching, or changing startup state on Reachy requires Chris's explicit final sign-off.
- Report only completion, a new blocker, deployment/recovery state, or a request that genuinely needs Chris.
- If no material state changed, stay quiet.
