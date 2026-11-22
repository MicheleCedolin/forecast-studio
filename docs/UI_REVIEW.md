# Forecast Studio UI review

Reviewer: ui-design-reviewer agent. Date: 2026-11-22. Scope: `src/tsfstudio/app/static/{index.html,app.js,app.css}`,
`server.py`, `service.py`, `models_info.py`.

## How this review was done

I ran the app for this review; it was not a code-only review. I started `python -m tsfstudio app --no-browser --port 8790` with
temporary data, runs and uploads folders, so the repository's `runs/` and `uploads/` were not touched. The data was two small
TSF files made with `tsfstudio.tsf.writer.write_tsf`:

- `demo_weekly`: 20 smooth weekly series, horizon 8.
- `demo_intermittent`: 20 intermittent monthly series, horizon 6.

I drove the app in a real browser through the whole flow:

- Data, then Model, then Settings, then Run, then Results, for **Theta on demo_weekly** and **Croston–SBA on demo_intermittent
  with rounding forced On**.
- Past runs and the two-run comparison.
- Layout at 1366×768, at 800×600 (where the note becomes a bottom panel), and at 375×812 (phone).
- Dark colour scheme (`prefers-color-scheme: dark`).

I checked 1920×1080 only in the code (`max-width: 1500px`, centred grid). I tested server and token errors from the code, not
by injecting failures. The server was stopped afterwards.

## Summary

The core flow is solid and unusually well thought through for researchers:

- The flow runs as one page in five sections.
- The pre-run check estimates time and memory.
- Long runs need a confirmation tick.
- There is a warning about zero forecasts.
- Model explanations sit in the margin with tags showing whether each citation was checked.
- Every run is reproducible through `manifest.json`.

A successful run takes 3 clicks after a dataset is chosen (model, Start run, and the scroll happens automatically).

The main problems are in four areas:

1. **Broken or missing paths on phone and after a reload.** On a phone the "Open" button for past runs is hidden. A run that is
   in progress is lost when the page is reloaded.
2. **Stale or ambiguous state.** Old results and a "Run finished" message stay on screen after the dataset or model changes.
3. **Accessibility.** The margin note is a live region, so a screen reader reads it again on every focus move. Every list row is
   its own tab stop. There is no dark mode.
4. **Number presentation.** Numbers use 4 significant digits, so the decimals in one column do not line up. There are no units,
   and sub-second runs show as "0:00".

Some research needs are only partly met. Seed, versions and parameters are not shown in the UI. Fallback details are sent by the
server but never displayed. The comparison view highlights "best" values even when the runs use different datasets.

---

## High

### H1. On a phone, past runs cannot be opened
- **Where:** `app.css:285` hides `.runrow > :nth-child(8)`, and child 8 is the "Open" button built at `app.js:869`.
- **What happens:** At widths of 760 px or less the only control that loads a past run has `display:none`. I confirmed this in
  the browser at 375 px: both Open buttons were hidden.
- **Why it matters:** On a phone, Past runs is a dead end. Results from a reload or an earlier session cannot be reached.
- **Fix:** Hide child 5 or 6 (SMAPE/MASE) instead of 8. Better still, make the whole run label a button that opens the run, and
  keep the checkbox as the only other control on the row.

### H2. Reloading during a run loses the job
- **Where:** `app.js:86-99` (`S.job` lives only in memory), `app.js:181-203` (init never checks for a running job),
  `service.py:130-138` (`system()` returns only `busy: true`).
- **What happens:** After a reload, a closed tab or a reopened browser during a long run, the UI shows a ready "Start run"
  button. There is no progress bar and no Cancel button. Pressing Start returns HTTP 409 as a 7-second toast.
- **Why it matters:** Runs of 12–30 minutes are expected (TBATS, ARIMA on NN5). The user cannot see or cancel a run they started,
  and may think the app has hung.
- **Fix:**
  - Return the current job's `view()` from `/api/system` (or add `GET /api/jobs/current`).
  - In `init()`, if a job is running, set `S.job` and call `pollJob()`.
  - Also show a "Run in progress: dataset/method" line in the masthead status.

### H3. The margin note is a polite live region, so a screen reader re-reads it on every hover or focus
- **Where:** `index.html:48` (`aria-live="polite"` on `<aside class="margin">`), `app.js:112-120` (`bindNote` replaces the note
  on every `mouseenter` and `focusin`).
- **What happens:** Tabbing through 43 datasets or 21 models makes a screen reader read a whole paragraph-long note for each row,
  including references. Mouse hovering does the same.
- **Why it matters:** Keyboard and screen-reader users effectively cannot use the lists.
- **Fix:**
  - Remove `aria-live` from the margin.
  - Link each row to a short description with `aria-describedby`, using `m.summary`.
  - Keep the full note visual-only, or add an explicit "Details" button that moves focus into the margin.

### H4. Old results and the old "Run finished" message stay after the selection changes
- **Where:** `app.js:642-644` (`renderProgress` keeps showing the last job), `app.js:677-708` (`renderSonuc` keeps `S.run`),
  `selectDataset` and `selectModel` at `app.js:309`/`453`, which never clear them.
- **What happens:** I ran Theta on demo_weekly, then picked demo_intermittent with Croston–SBA. The Run step still said "Run
  finished (0:00). Results are below." The Results section still showed the demo_weekly/Theta run. The only clue was the
  highlighted title.
- **Why it matters:** A researcher can easily read or download the wrong run's metrics.
- **Fix:**
  - When the config built by `buildConfig()` differs from the config of the displayed run, show a banner above Results: "These
    results are from *demo_weekly, theta (Python)*, not from the current selection." Add a "Clear" link.
  - Clear the finished-job message whenever the selection changes.

### H5. No dark mode; colours are hard-coded in JS
- **Where:** `app.css:12-31` (light tokens only, no `prefers-color-scheme` block). Hex colours are written directly in JS at
  `app.js:328` (profile palette), `368` (sparkline), `755-758` (legend), `777-790` (chart), `824` and `828` (histogram).
- **What happens:** With the OS in dark mode the page stays bright `#f5f7f8`. I confirmed this in the browser. Charts could not
  follow a theme even if CSS added one.
- **Why it matters:** The review brief asks for dark mode. Long sessions on a bright page are tiring, and the app looks broken
  next to dark OS chrome.
- **Fix:**
  - Add `@media (prefers-color-scheme: dark) { :root { ... } }` with dark values for `--paper`, `--white`, `--ink`, `--rule`
    and the others.
  - Keep the highlighter yellow, but use a darker amber for `--hl-soft`.
  - Replace the hex colours in JS with `var(--ink)` and similar (SVG `stroke`/`fill` accept `var()` through `style`), or with
    CSS classes (`.chart .hist-line`, `.chart .fc`, ...).
  - Add `color-scheme: light dark`.

---

## Medium

### M1. Number formatting is inconsistent and has no units
- **Where:** `app.js:38,42` (`maximumSignificantDigits: 4` for every metric), `app.js:720-722` (summary table),
  `app.js:844` (error table), `app.js:701` (mean forecast / actual), `models_info.py:341-358` (units exist but are never shown
  in tables).
- **What happens:**
  - In one column, values look like `0.05388 / 1.066 / 3.591 / 257.9`.
  - In the error table, values look like `4.2`, `3.25`, `1.1`, so the decimal points do not line up even with `tabular-nums`.
  - WAPE has the unit "percent" but is shown as `5.291` with no `%`.
  - "Mean forecast / actual" shows `0.95 / 0.6917`.
- **Why it matters:** Numbers are hard to scan and compare. It is unclear whether a WAPE of 257.9 is a ratio or a percentage.
- **Fix:**
  - Use a fixed number of decimals per metric, held in `METRICS_INFO` (for example SMAPE/mSMAPE/MASE/nRMSE to 3, MAE/RMSE to
    3 significant digits with thousands separators, WAPE to 1 with a `%` suffix).
  - Use `minimumFractionDigits` equal to `maximumFractionDigits` so decimals line up.
  - Add a "Unit" column to the Summary table, or put the unit in the header (`WAPE (%)`).

### M2. Run time shows "0:00" for short runs
- **Where:** `app.js:631` (`mmss`), used at `644`, `690` and `868`.
- **What happens:** Every test run showed `0:00` in the progress line, in Results and in Past runs.
- **Why it matters:** Run time matters when comparing methods (it is a research output), and "0:00" looks like a failure.
- **Fix:** Below 60 s show `0.4 s` or `12.3 s`; from 60 s show `m:ss`.

### M3. Fallback details are never shown
- **Where:** `service.py:587` returns `fallbacks` (up to 30 rows). `app.js:691` shows only `m.fallbacks.count`.
- **Why it matters:** The brief lists fallback-to-SNaive as a key state. A non-zero count gives no way to see which series fell
  back or why, without opening `fallbacks.csv`.
- **Fix:**
  - When the count is above 0, give the fact the `flag` style.
  - Add a `<details>` below it listing series name, reason and replacement method, using `d.fallbacks`.
  - Link to `fallbacks.csv`.

### M4. Reproducibility information is not visible in the UI
- **Where:** `app.js:687-706` (Results shows only series, horizon, run time, fallbacks and rounding). Past runs at
  `app.js:852-877` has no "reuse settings" action.
- **What happens:** Seed, workers, missing-value policy, model parameters, package and engine versions, and the effective lag
  are only in `manifest.json`.
- **Why it matters:** Researchers need to check what produced a table without opening JSON. They also cannot start again from a
  past configuration.
- **Fix:**
  - Add a collapsible "Run configuration" block in Results, read from `manifest.config`, `manifest.engine` and the versions
    stored in the manifest.
  - Add "Load these settings into the workspace" in Results and in each Past runs row. It should set `S.sel`, `S.settings` and
    `S.params` from `manifest.config`.

### M5. The comparison view can mislead
- **Where:** `app.js:878-902`.
- **What happens:**
  - The "best" highlight (`884`, `897`) is applied even when the warning at `891` says the datasets differ. I saw this with
    SMAPE 1.701 (intermittent) against 0.054 (weekly).
  - Bar labels (`901`) show only the method, so two runs of the same method on different data cannot be told apart.
  - The table needed horizontal scrolling at 1366 px.
  - There is no ranking across runs.
- **Why it matters:** The brief asks whether results can be compared across models and metrics. At the moment cross-dataset
  numbers are presented as rankable.
- **Fix:**
  - Suppress the `best` highlight and the bar highlight when `sameData` or the horizon check fails.
  - Use the full run label in the bars.
  - Add a "Mean rank" column across the selected metrics. Later, consider a critical-difference style view when 3 or more
    methods were run on the same dataset.

### M6. Hover looks stronger than selection, and disabled rows react to hover
- **Where:** `app.css:140-141` and `178-179` (hover `--hl-soft #fff1a3` is darker than selected `#fffbe0`), `app.css:140` and
  `145` (`.drow.dim` still gets the hover background), `app.css:212` and `216` (table row hover uses the same yellow as
  `td.best`).
- **What happens:**
  - With the pointer resting on a row, that row looks more selected than the actual selection.
  - A "file missing" dataset turned yellow on hover in the browser.
  - In the comparison table, the hovered row looked as if every cell were "best".
- **Fix:**
  - Use a neutral hover (`--rule-soft` or a 4 % ink tint).
  - Keep yellow for selection and best values only.
  - Add `.drow.dim:hover { background: transparent; }`.
  - Make the selected state stronger: a thicker left bar plus a check mark or "Selected" text for non-colour users.

### M7. Keyboard model for the lists
- **Where:**
  - `app.js:287` and `427`: every `role="radio"` row has `tabindex=0`, with no roving tabindex and no arrow keys.
  - `app.js:269`: `role="tab"` buttons have no `tabpanel` and no arrow keys.
  - `app.js:835-837`: sortable `th` responds only to Enter.
  - `app.js:843`: error rows are clickable `tr` elements.
- **What happens:** About 64 tab stops must be crossed before reaching Settings.
- **Why it matters:** Keyboard use is slow, and the radiogroup does not behave as screen-reader users expect.
- **Fix:**
  - Use a roving tabindex: only the checked row, or the first row, gets `0`, and Up/Down move focus and selection.
  - Make the dataset filter tabs plain toggle buttons (`aria-pressed`), or implement the full tabs pattern.
  - Put a `<button>` inside each sortable `th`.
  - Put a link or button on the series name instead of making the `tr` focusable.

### M8. Progress is not announced well, and errors during polling are left unhandled
- **Where:**
  - `app.js:638`: the whole progress block, `role="status"`, is rebuilt every 600 ms, so a screen reader may announce it
    constantly. The bar is `aria-hidden`.
  - `app.js:628`: a failed poll shows a toast and **stops polling for good**, leaving "running" and Cancel on screen.
  - `app.js:641`: the cancel promise has no `catch`.
- **Fix:**
  - Use `<progress>` or `role="progressbar"` with `aria-valuenow` and `aria-valuemax`. Update the text node in place, and
    announce only stage changes.
  - On a poll error, retry with backoff 3–5 times. Then show a persistent `note err`: "Lost contact with the server. Is the
    command window still open?" with a "Retry" button.
  - Add a `.catch(e => toast(e.message, true))` to the cancel call.

### M9. Phone layout problems beyond H1
- **Where:**
  - `app.js:283` and `app.css:280`: header spans have no class `d`, so on a phone the header still shows "Domain", which
    wraps above "Status". I confirmed this in the browser (header: Name, Domain, Status; rows: name, status).
  - `app.css:270-276`: the masthead, steps and Quit take about 400 px before any content.
  - `app.js:746-753`: the chart toolbar wraps to about 190 px tall.
  - The bottom-sheet note takes 30vh.
- **Fix:**
  - Add `class: "d"` to the header Domain span, and hide the header Frequency/Series cells with explicit classes rather than
    `nth-of-type`.
  - On a phone, make the masthead compact (hide the subtitle, keep the engine status in one line).
  - Move "Quit application" to the end of the page.
  - Turn the three "highest / median / lowest error" buttons into one `<select>`.

### M10. Turkish strings and CLI wording in an English UI
- **Where:**
  - `app.js:161-162`: "Birim:" and "Dikkat." in `metricNote`.
  - `server.py:114`: "Beklenmeyen hata".
  - `service.py:603`: "Bilinmeyen metrik."
  - `server.py:169`: the template downloads as `forecast_studio_sablon.xlsx`, while `app.js:380` says
    `forecast_studio_template.xlsx`.
  - The runner warning shown in both Run and Results says "run with --integer-conversion off", which is a CLI flag the UI user
    cannot type, and it repeats the UI's own note (`app.js:491`).
- **Fix:**
  - Translate these strings: "Unit:", "Caution.", "Unexpected error", "Unknown metric.", `forecast_studio_template.xlsx`.
  - Have the runner return a warning code, so the UI can show "Set *Round to integers* to Off (Settings)" with a button that does
    it.
  - Remove the duplicate.

### M11. Text written for the author's study also appears for uploads
- **Where:** `app.js:500` ("Defaults are the values from your own study"), `app.js:503` ("(your study)"), `app.js:447`.
- **Why it matters:** For uploaded data there is no study, so the text is wrong. It also makes the app read as a personal tool,
  which matters if it is shared with other researchers.
- **Fix:** Choose the wording by `d.kind`: for catalogue data "Defaults come from the reference study (catalogue)"; for uploads
  "Defaults come from the file header".

### M12. Large uploads show no progress
- **Where:** `app.js:392-406`. `fetch` with a body offers no progress, and the only feedback is "Uploading x…". The limit is
  400 MB (`service.py:41`).
- **Fix:**
  - Use `XMLHttpRequest` with `upload.onprogress` and reuse the `.track` bar.
  - Disable "Choose file" while an upload runs.
  - Show the file size before uploading.

---

## Low

- **L1. Information lost on reload.** Results are not restored after a reload (`S.run` is not stored). Store
  `store.set("lastRun", id)` and call `loadRun` in `init`. (`app.js:197-202`)
- **L2. Uploaded datasets show "—" for Series** in the list (`service.py:218`, where `n_series` is `None`). The header reader
  could count data lines cheaply, or the value could be filled in after the first `dataset_detail`.
- **L3. Hover note replaces the selected model's note.** After `selectModel` scrolls with `scrollIntoView`, the resting pointer
  lands on another row and the margin shows that model. I saw this: Theta was selected while the note showed Pooled regression.
  Ignore `mouseenter` for about 400 ms after a programmatic scroll, or show a small "Selected: Theta" header above hover notes.
  (`app.js:113`, `467`)
- **L4. The rail shows the method id, not the name** ("theta", "croston_sba"). Use `m.name`. (`app.js:241`, `648`)
- **L5. Histogram labels overlap.** The mean and median labels collide when the values are close (`app.js:827-828`); I saw this
  with 1.066 and 1.041. Place them on opposite sides of the line, or into a legend line under the chart.
- **L6. Charts are mouse-only.** The readout works only with the mouse (`app.js:803`). Add `tabindex=0` on the SVG with
  Left/Right keys to move the crosshair, and a visually hidden data table, or a "Download this series (CSV)" link.
- **L7. Step links in the rail do nothing in Past runs**, because the sections are hidden (`app.js:243-249`,
  `index.html:23-27`). A step click should switch to the workspace first.
- **L8. "Measure time (8 series)" is disabled without a visible reason** for global models (only a `title`) (`app.js:604-608`).
  Add a hint: "Available for local methods only".
- **L9. Disabled "Start run" gives no reason** (`app.js:603`). Add a one-line reason under the button: "Tick the confirmation
  box" or "Choose at least one measure".
- **L10. Placeholder contrast.** `#7e8ea0` on white (`app.css:116`) is about 3.4:1, below AA. The placeholder carries real
  information ("8 (data)", "all"). Darken it to `--pencil`, or show the effective default as helper text under the field.
- **L11. The Files list has no explanation and no zip download** (`app.js:705-706`). Add a one-line description per file
  (forecasts, actuals, metrics workbook, manifest) and a "Download all (.zip)" endpoint.
- **L12. Full paths are shown.** The Past runs lead shows the full absolute runs path (`app.js:873`), which is long, wraps
  badly and exposes the user name in screenshots. Show only the folder name, with the full path in `title`.
- **L13. The token error shows only as a toast.** A token or host rejection (`server.py:89`, `99`) appears as a toast after
  every action. When a 403 with the token message arrives, show a persistent banner with a "Reload page" button.
- **L14. Unclear status dot.** The yellow memory dot (`app.css:51`) has no text meaning. At 0.6 GB free the dot is yellow but
  nothing says "low". Add "(low)" to the text.
- **L15. Unexplained engine badges.** The "Py/R" badges with strike-through or red outline have no legend. Add one line under
  the Model lead: "Py / R = engine; struck through = not offered; red = not installed". (`app.js:409-414`, `app.css:184-186`)

---

## What works well (keep)

- The pre-run check, the long-run confirmation and the memory block appear before the expensive action.
- Defaults are visible: changed fields are marked, "back to default" is offered, and the rounding auto-off for the Croston family
  is explained in place.
- The zero-forecast warning and the forecast diagnostics address a real problem with metrics on intermittent demand.
- Tags show which citations were checked against a record and which were not.
- Typography and hierarchy are consistent, the masthead and steps read like a paper, and `prefers-reduced-motion` is respected.
- Data is written to the page only through `textContent`, the API is protected by a token, and downloads are limited to the
  outputs listed in the manifest.

---

## Top 5 changes to make first

1. **Reconnect to a running job and fix the phone dead end** (H2, H1). Return the current job from `/api/system` and resume
   polling in `init()`. Stop hiding the "Open" column on phones.
2. **Mark or clear stale results** (H4). Show a banner when the displayed run does not match the current selection, and clear the
   "Run finished" message when the selection changes.
3. **Make numbers readable for research** (M1, M2, M3). Use fixed decimals per metric with units and `%` for WAPE, show
   seconds for short runs, and list fallback series.
4. **Accessibility pass** (H3, M7, M8). Remove `aria-live` from the margin, use a roving tabindex with arrow keys in the
   dataset and model lists, add a real progressbar, and retry polling with a persistent error.
5. **Dark mode and colour tokens** (H5, M6). Move all chart colours from JS into CSS variables, add a
   `prefers-color-scheme: dark` palette, and separate hover (neutral) from selection and best (yellow).
