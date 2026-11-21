# llmtuner

A small eval harness that any project or agent can report to.

You describe a project in one file: which commands to test, which cases to test them on, and how to grade the answers. llmtuner runs the cases, stores every result in one local database, and reports four numbers:

1. **Pass rate**, with its 95% interval (Wilson score interval).
2. **Grader vs human**: how well each grader agrees with your own labels.
3. **Detection floor**: the smallest change this many cases can show. At least 6 cases must change one way before two runs differ measurably, so 12 cases can only show a 50-point change and 60 cases a 10-point one.
4. **Got worse**: which cases passed on the baseline and fail now.

A result is only as good as the weakest of the four. A pass rate from an unchecked grader, or a 2-point "improvement" on 20 cases, tells you nothing.

The core uses only the Python standard library (Python 3.12 or later).

## Quick start

The example sorts support tickets with a rule-based stand-in for a model, so it runs offline and costs nothing.

```bash
cd examples/support
export PYTHONPATH=../../src            # or install the package, see below
python3 -m llmtuner run triage-v1 tickets --set-baseline
python3 -m llmtuner run triage-v2 tickets
python3 -m llmtuner compare 1 2
python3 -m llmtuner show 2 --failed
```

The run ids 1 and 2 assume a fresh store; `llmtuner runs` shows the real ids.

To get an `llmtuner` command everywhere:

```bash
uv tool install --editable /path/to/LLMtuner
```

## How a project connects

Put an `llmtuner.toml` in the project. See [examples/support/llmtuner.toml](examples/support/llmtuner.toml) for a commented one.

```toml
project = "my-app"

[targets.current]
command = ["python3", "answer.py"]     # reads one input on stdin, prints the answer
timeout = 60

[datasets.tickets]
path = "tickets.jsonl"

[graders.category]
type = "exact"
```

**Targets** are commands. Anything that reads stdin and prints an answer fits: a script, an HTTP call through `curl`, or an agent CLI such as `claude -p --model haiku "..."`. Commands are lists of arguments and never go through a shell. A timeout or a crash counts as a failed case; the run carries on.

**Datasets** are JSON Lines, one case per line:

```json
{"id": "t1", "input": "Where is my order?", "expected": "shipping", "split": "test"}
```

`split` is `dev` or `test` and defaults to `test`. Tune prompts and graders on `dev`; report on `test`, which is what `run` uses unless you pass `--split`. `expected` may be an object when graders need several fields, and `context` holds text the system was given, which judge graders see.

**Graders** decide pass or fail. A case passes when every required grader passes.

| type | passes when | options |
| --- | --- | --- |
| `exact` | the output equals the expected value | `field`, `ignore_case` (default true) |
| `contains_all` | every listed phrase appears | `values`, or `field` to read them from each case |
| `not_contains` | none of the phrases appear | `values` |
| `regex` | the pattern matches | `pattern` |
| `max_words` | the output is short enough | `limit` |
| `nonempty` | the output is not blank | |
| `json` | the output parses as JSON | `required_keys` |
| `json_match` | every field in the case's `expected` object equals the same field of the JSON output; extra output fields are ignored; lists must match in full, including objects inside them | `field`, `ordered` (default false: list order is ignored at every depth) |
| `judge` | a model answers a yes/no question the right way | `command`, `question`, `fail_on` (default `"yes"`), `timeout`, `include_expected` |

Add `required = false` to report a grader without letting it decide pass or fail. A dataset can pick its graders with `graders = ["a", "b"]`; otherwise it uses all of them.

## Real models through OpenRouter

The package ships one ready-made target: an adapter for [OpenRouter](https://openrouter.ai), which also works with other OpenAI-compatible APIs through `--base-url`. Each model becomes a target:

```toml
[targets.nemo]
command = ["{python}", "-m", "llmtuner.adapters.openrouter", "--model", "mistralai/mistral-nemo",
           "--system-file", "triage_prompt.txt", "--max-tokens", "16"]
model = "mistralai/mistral-nemo"
timeout = 120
```

`{python}` stands for the Python that is running llmtuner, so the adapter can always be imported however llmtuner was installed. The same command works as a judge grader's `command`.

The key is read from the `OPENROUTER_API_KEY` environment variable, else from a `.env` file in the project folder or one above it, else from `~/.config/llmtuner/env`. A `.env` line looks like `OPENROUTER_API_KEY='...'`. The key is never taken from the command line or the config, so it is never stored with a run. Keep `.env` out of git.

Each case is one paid API call, plus one per judge grader. The adapter never follows a redirect, and `--timeout` is one deadline for the whole call, retries included. For an `http://` base URL it ignores proxy settings from the environment; `https://` keeps them.

## Cost and ranking

A target can report what a call cost. llmtuner sets the `LLMTUNER_META` environment variable to a file path for each call; a target that writes a JSON object there with any of `cost`, `cost_unit`, `prompt_tokens`, `completion_tokens` and `model` gets those stored with the case. The OpenRouter adapter does this, in OpenRouter credits. Calls made by judge graders are not counted yet.

The report then shows total cost, cost per passed case, tokens and the median time per case. Anything a target did not report is shown as "not reported", never as zero.

```bash
llmtuner rank tickets
```

`rank` compares only runs on the same dataset file, with the same graders, over the same set of cases. The newest run decides which dataset file and graders count as current, and among runs on those the largest set of cases is used, so a quick `--limit` run does not hide a full one. It takes the newest run of each target in that group and orders them by pass rate, then by cost per passed case, but only when every row reports cost in one unit; otherwise it says so and orders by pass rate. Each row says whether it differs measurably from the leader, so a higher place alone is not read as "better". A target with no run in that group (an older dataset, other graders, or a `--limit` run) is listed as skipped with the reason.

## Repeated trials

A model can answer the same input differently each time, so one run of a case may say little. Run every case several times:

```bash
llmtuner run current tickets --trials 5
```

Each trial is stored as an ordinary run, and the runs form one group. The report then shows three figures, and says which is which:

- **Passed all trials**: cases that passed every time. This is what `compare`, `gate` and `rank` use for a group, so they measure reliability.
- **Passed at least once**: cases the target can get right.
- **Per-trial pass rate**: passes over all case results (one case in one trial: 3 cases and 5 trials give 15). Its interval counts each case once, so repeating a run never makes a result look more certain than its number of cases allows.

Cases whose trials disagree are listed as unstable. A trial that ended in a target error, such as a timeout, counts as a failed case result in all three figures, so its case does not count as having passed every trial. It is also listed separately under target errors, and by itself it does not make a case unstable.

`compare` between two groups also tests each case on its own: trials passed before and after, with a Fisher exact p-value. That can show a change on a single case, which the overall verdict cannot do with fewer than 6 cases. With 5 trials a side it only detects near-total swings: 0 of 5 against 5 of 5 gives p = 0.008, 1 against 5 gives 0.048, and 2 against 5 gives 0.17. When many cases are tested one by one, about 1 in 20 crosses 0.05 by chance. The test treats trials as independent attempts, so for a target that always gives the same answer, repeating adds no information even though the per-case p-value shrinks with more trials; trust it only for targets whose answers actually vary.

`gate` fails a group whose trials did not all finish, on either side of the comparison, instead of passing it on partial results.

Trials multiply the number of target calls and the cost. Passing every trial is stricter with more trials, so compare groups with the same number; `compare` warns when they differ, and `rank` only ranks groups with the same number of trials. A target can read which trial it is from the `LLMTUNER_TRIAL` environment variable.

## The web page

```bash
llmtuner ui
```

This starts a page on `http://127.0.0.1:8765` and opens it in your browser (`--port` to change the port, `--no-open` to skip the browser). It shows the same store the commands write to:

- **Runs**: every run of a project with its pass rate and interval; tick two to compare them.
- **Run**: the four numbers, usage, graders, and each case with its input, output and grades.
- **Compare**: the difference between two runs, the verdict, and the cases that got worse or better.
- **Ranking**: the targets of a project on one dataset, by pass rate and cost per passed case.
- **Label**: one output at a time with Pass, Fail and Skip (keys `p`, `f`, `s`). The grader's verdict is not sent to the page, so your label stays independent.

The page is served to this machine only, takes no data from the network, and treats everything a model wrote as plain text. It needs nothing beyond the standard library.

## Checking a judge

A judge grader is a model, so it can be wrong in the same ways as the system it grades. Before trusting its numbers, label some outputs yourself:

```bash
llmtuner label 7 --grader polite      # shows each output, hides the judge's verdict
llmtuner report 7                     # now shows agreement, kappa and catch rate
```

Kappa corrects agreement for chance. The report treats 0.6 as enough to gate a release and 0.8 as enough to run unsupervised, and says so in words. It will not rate a grader on fewer than 20 labels. `llmtuner label` without `--grader` records your verdict on the whole case, which checks the combined result of all graders.

## Commands

| command | what it does |
| --- | --- |
| `info` | show this project's targets, datasets and graders |
| `run TARGET DATASET` | run, grade, store and report; `--jobs N` runs cases in parallel, `--trials K` repeats every case |
| `report [RUN]` | the four numbers for a run (default: this project's latest) |
| `compare BASELINE CANDIDATE` | case-by-case comparison of two runs |
| `baseline RUN` | make a run the baseline for its dataset |
| `gate [RUN]` | exit 1 if the pass rate fell more than `--max-drop` below the baseline |
| `show [RUN]` | read inputs, outputs and grades; `--failed` for failures only |
| `label [RUN]` | record human pass/fail labels |
| `runs` | list recorded runs; `--all` for every project |
| `rank DATASET` | order this project's targets by pass rate and cost per passed case |
| `ui` | open the local web page for browsing runs and labelling outputs |
| `mcp` | start the MCP server |

Every command takes `--json` for machine-readable output, `--config` to point at another project, and `--store` to use another database.

Results live in one SQLite file for all projects. The store folder is `$LLMTUNER_HOME`, else `$XDG_DATA_HOME/llmtuner`, else `~/.local/share/llmtuner`; the file is `store.db` inside it.

## In CI

```bash
llmtuner run current tickets && llmtuner gate --max-drop 0.03
```

A baseline must be set once (`llmtuner baseline RUN`, or `run --set-baseline`), or the gate fails with "no baseline set".

The gate is a tripwire for large regressions. With few cases it cannot see small ones, and it prints the size it can see.

## How an agent connects

`llmtuner mcp` starts an MCP server on stdio. It needs the `mcp` package (version 2), which the rest of llmtuner does not. This command fetches it on the fly and leaves the project folder untouched:

```bash
uv run --no-project --with 'mcp>=2,<3' --with-editable /path/to/LLMtuner llmtuner mcp
```

Give that command to the agent as an stdio MCP server. In Claude Code:

```bash
claude mcp add llmtuner -- uv run --no-project --with 'mcp>=2,<3' --with-editable /path/to/LLMtuner llmtuner mcp
```

Under this command, `python3` inside a target resolves to uv's temporary environment, not your project's. If a target needs your project's packages, name its interpreter in the target command, for example `[".venv/bin/python", "answer.py"]`.

The agent gets seven tools: `describe_project`, `run_eval`, `get_report`, `compare_runs`, `check_gate`, `list_runs` and `show_cases`.

`run_eval` runs whatever targets the named project's `llmtuner.toml` defines, so an agent that can write that file can choose what runs; connect it only to agents you would let run commands.

Two things are left out on purpose. Tools take the *name* of a target from the project's config, never a command to run. And agents cannot write human labels or set the baseline: those are the ground truth the graders are checked against.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

All tests run offline against stub scripts. The MCP test is skipped unless `mcp` is installed; to include it:

```bash
PYTHONPATH=src uv run --no-project --with 'mcp>=2,<3' python -m unittest discover -s tests
```

## Not in this version

Trace collection, panels of several judges, hallucination detectors, monitoring of live traffic, and counting the cost of judge calls.
