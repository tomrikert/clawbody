# Native Reachy App Contract Review

Reviewed against the installed Reachy Mini 1.9 SDK and its bundled
`reachy-mini-app-assistant` validator.

## Confirmed contract

- The project name is `clawbody`; the validator derives the package from the
  entry point and therefore expects class `ReachyMiniOpenclaw` in
  `reachy_mini_openclaw.main`.
- The native class must explicitly inherit `reachy_mini.ReachyMiniApp`.
- The entry point must be:

  ```toml
  [project.entry-points."reachy_mini_apps"]
  clawbody = "reachy_mini_openclaw.main:ReachyMiniOpenclaw"
  ```

- The SDK owns the `ReachyMini` context, creates a `threading.Event`, and calls
  `run(reachy_mini, stop_event)` through `ReachyMiniApp.wrapped_run()`.
- The app must stop when that event is set and must not close/disconnect the
  SDK-owned robot itself.
- The existing README frontmatter, root `index.html`, root `style.css`, and src
  package layout satisfy the validator's static metadata/layout requirements.
- The official validator finishes by installing the project into an isolated
  venv, verifying the entry point, and uninstalling it.

## Current defects to correct

1. `ClawBodyApp` neither has the validator-required name nor inherits
   `ReachyMiniApp`. The current validator fails before installation with the
   expected entry point `reachy_mini_openclaw.main:ReachyMiniOpenclaw`.
2. The native wrapper catches and suppresses every exception. That prevents the
   app manager from observing a failed app and undermines watchdog/restart
   behavior. Cleanup should occur in `finally`, then the failure must propagate.
3. The external stop event is only polled by the record/play loops. The
   conversation handler remains in the `asyncio.gather()` set, so a dashboard
   stop can leave `run()` waiting indefinitely. A stop watcher must cancel all
   app tasks promptly.
4. `ClawBodyCore.stop()` is synchronous but tries to run the asynchronous
   gateway disconnect with `run_until_complete()` on the current event loop.
   When called from the native wrapper after `run_until_complete()`, this is
   fragile; when called while a loop is active, it is invalid. Async shutdown
   needs a single owning loop and an idempotent lifecycle.
5. Configuration validation calls `sys.exit(1)` inside an embeddable app class.
   Native lifecycle errors should raise typed exceptions so the SDK can record
   `app.error` and manage failure correctly.
6. Native camera tracking currently builds its daemon URL from
   `ROBOT_HOST`/`ROBOT_PORT`. On the Wireless body, the SDK-owned robot is local
   and the daemon-side HTTP API should use localhost (or an explicit native
   daemon URL), not the remote workstation default.
7. The package deliberately depends on remote xeon STT/TTS and the existing
   OpenClaw gateway. Startup must tolerate those services being temporarily
   unavailable and reconnect with bounded backoff rather than exiting or
   pretending the app is healthy.

## Required implementation shape

- Introduce `ReachyMiniOpenclaw(ReachyMiniApp)` with the exact entry point.
- Keep `ClawBodyCore` reusable for CLI/simulator operation.
- Give the core an idempotent async shutdown path and an explicit external-stop
  watcher.
- Let fatal native exceptions propagate after cleanup.
- Add contract tests for inheritance, entry-point loading, stop behavior,
  cleanup idempotency, and reconnect/backoff behavior.
- Preserve the existing `main` agent and `agent:main:reachy` session. Native
  packaging changes transport/lifecycle only; it must not create Asmo(deux).
