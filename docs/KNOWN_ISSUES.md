# Known issues

Tracked defects and limitations that are understood but not yet fixed. Each
has an ID that the code and UI reference directly, so a warning a user sees on
screen can be traced to a written explanation.

| ID | Title | Severity | Blocks |
|---|---|---|---|
| [GSL-1](#gsl-1) | Ground all timing-dependent analysis in measured presentation timestamps | Largely resolved | — |
| [GSL-2](#gsl-2) | Push-triggered CI stopped creating runs (GitHub Actions incident) | Resolved | — |

---

## GSL-1

### Ground all timing-dependent analysis in measured presentation timestamps

**Status:** largely resolved · **Tracked at:**
<https://github.com/johnsonto07/golf-swing-lab/issues/1> ·
**Originally filed:** 2026-08-06 · **Corrected:** 2026-08-07

### The original diagnosis was wrong, and this is what it got wrong

This issue was filed as "preview frames do not map back to source frames on
VFR video", based on a real phone clip that appeared to lose 46 frames and drift
~9% in time. **Both symptoms were artefacts of trusting container metadata.**

Measuring the media directly:

| | Container claimed | Actually measured |
|---|---|---|
| Frames | 484 | **438 decode** |
| Frame rate | 22.873 fps average | **25.000 fps, every gap exactly 0.04 s** |

The clip is constant frame rate. `nb_frames` was wrong, and `avg_frame_rate` —
being frames ÷ duration — inherited that error. Nothing was dropped, and no
drift existed; the 9% figure was the gap between a true 25 fps and a fabricated
22.873.

The preview pipeline was also exonerated. Running the project's own
`generate_preview` on a **genuinely** variable-frame-rate fixture (gaps
alternating 0.033 s and 0.067 s) returned 60 frames from 60 with a maximum
timestamp drift of **0.00000 s**, because `-fps_mode passthrough` carries
original timestamps through unchanged.

**So preview frame N is source frame N** for the tested pipeline, and no
separate frame mapping is required.

### What was actually broken

Not the mapping — the *source of truth*. Timing was computed as
`frame_index / nominal_fps`, and the frame count came from the container. Both
are claims; the frame timestamps are the evidence.

### Acceptance criteria

- [x] Per-frame presentation timestamps measured and persisted (`timeline.json`)
- [x] Decoded frame count is the source of truth, never `nb_frames`
- [x] CFR/VFR classified from measured spacing, never from metadata disagreement
- [x] Explicit timing trust states: measured, partially interpolated, nominal
      only, unavailable
- [x] Nominal-only timing refuses durations and tempo rather than estimating
- [x] Durations and tempo computed from measured timestamps
- [ ] Frame identity re-verified for any future pipeline that re-encodes
      without `-fps_mode passthrough`

### What remains

Media where **no** timestamps can be read still cannot support durations or
tempo. That is not a defect to fix — it is a limit of the input, and the app
refuses rather than estimating. The issue stays open only for the last box:
frame identity is verified for the current passthrough pipeline, and any future
change that re-encodes differently must re-establish it rather than inherit the
assumption.

---

## GSL-2

### Push-triggered CI stopped creating runs

**Status:** resolved (platform-side) · **Severity:** medium — CI silently
stopped running on push while `workflow_dispatch` kept working, so the workflow
looked healthy while the badge could go stale unnoticed.

### Symptom

For roughly a three-hour window, pushes created no workflow run.
`workflow_dispatch` continued to work, so nothing in the Actions UI looked
wrong.

### What was ruled out

Every item below was checked and found correct, so none was the cause:

| Checked | Result |
|---|---|
| Workflow contents and parsed triggers at the pushed commit | correct — `on.push.branches` matched the pushed branch |
| Workflow present on the default branch | yes, byte-identical to local (no BOM, LF endings) |
| Path / event filters | none that could exclude the commits |
| Actions enabled for the repository | `enabled: true`, `allowed_actions: all` |
| Rulesets / branch protection | none configured at the time |
| Commit messages | no `[skip ci]`-style directives |
| Push credential | HTTPS via Git Credential Manager as a real user — **not** `GITHUB_TOKEN`, not a GitHub App — and the same credential had triggered runs minutes earlier |
| Hidden / skipped / deleted runs | Actions API `total_count` accounted for every run |
| Workflow state via API | `active` throughout |

### A hypothesis that was tested and disproved

The obvious suspect was a stale workflow registration left by the
`master` → `main` rename: the workflow's `created_at` and `updated_at` were
both still the moment it was first registered from a `master` push, despite the
file changing on `main` since.

**Three remedies were applied and none restored push triggering:**

1. Disabling and re-enabling the workflow (this did move `updated_at`).
2. Re-saving the trigger block in materially equivalent form.
3. Renaming `tests.yml` → `ci.yml`, which produced a genuinely new workflow id.

And a controlled experiment killed the theory outright: a push to a brand-new
`ci-diagnostic` branch, explicitly listed in the workflow's push filter, was
**also** ignored. The fault was never specific to `main`, and never about the
deleted branch.

### Actual cause

A transient GitHub Actions incident affecting **event-to-run creation**, not
this repository's configuration. Three observations fix it as platform-side:

- GitHub's own managed dependency-graph workflow also stopped firing on pushes
  over the same window.
- Registration kept working throughout — pushing `ci.yml` created a new
  workflow id — so pushes were being processed for some purposes but were not
  producing runs.
- Job scheduling degraded as well: a dispatched run sat queued for over ten
  minutes and its three jobs were then cancelled without ever starting.

Actions later recovered on its own: a dispatch completed in 80 seconds with all
three Python jobs green.

### If it recurs

Do not start by rewriting workflow files. Check in this order:

1. Does `workflow_dispatch` work? If yes, the file and registration are fine.
2. Do other integrations create check suites for the pushed commit while
   `github-actions` creates none? That isolates it to Actions.
3. Do dispatched jobs start promptly, or sit queued? A queue stall indicates a
   platform incident rather than anything in the repository.
4. Check <https://www.githubstatus.com/> before changing anything.

The repository was left on `ci.yml`. The rename was unnecessary in hindsight but
is harmless, and reverting it would churn the workflow id again for no benefit.
