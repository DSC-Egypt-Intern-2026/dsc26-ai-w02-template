# W2 task · Rawaj Pay: audit the extract, then choose the threshold

DSC Summer Internship 2026 · AI & Deep Tech track · Week 2 · expected time 2–3 hours

## Scenario

Rawaj Pay's data team has sent a two-month extract of instant transfers with a mule label.
Dina, Head of Fraud, wants two answers before anyone trains a model: **can this data be
trusted**, and once there is a score, **where should the line be drawn** so her five analysts
are not buried? You will write a small, well-tested library that answers both. No model
training is needed: the grader feeds your functions data and scores.

All data in this task is synthetic.

## Set up

1. On GitHub, open this starter (`dsc26-ai-w02-starter`) and press **Use this template → Create
   a new repository**. Owner: your own account. Name: exactly **`dsc26-ai-w02`** (lowercase).
   Public is recommended; if private, add the DSC grader account named on the task page as a
   read-only collaborator.
2. Clone it, then:

   ```bash
   uv sync            # installs Python 3.12 packages pinned in pyproject.toml / uv.lock
   uv run pytest      # the 5 public tests fail until you write the code; that is expected
   ```

Versions: Python 3.12, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1 (scipy 1.18.1 comes with
it), pytest 9.1.1. **No other dependencies**: the grader has nothing else installed.

## Layout

```
dsc26-ai-w02/
├── pyproject.toml              # provided, do not change the dependencies
├── uv.lock                     # provided
├── rawaj/
│   ├── __init__.py
│   ├── audit.py                # you write
│   └── metrics.py              # you write
├── data/sample_transfers.csv   # provided, 2,000 synthetic rows to play with
├── tests/test_public.py        # provided
├── FRAMING.md                  # you write (template provided)
└── AI_USAGE.md                 # you write (template provided)
```

Keep the file names, the package name `rawaj`, and every function name and signature exactly
as they are in the starter. The grader imports them by name.

## What to build

### `rawaj/audit.py`

```python
def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict: ...
```

Raises `KeyError` if any of the three named columns is missing. Returns a dict with exactly
these keys:

- `n_rows` (int).
- `missing`: `{column: fraction}` for **every** column. Missing means NaN/None, or a string that
  is empty or only whitespace (spaces, tabs, newlines).
- `duplicate_ids`: sorted list of non-missing `id_col` values that appear more than once.
- `duplicate_rows` (int): rows that exactly repeat an earlier row.
- `constant_columns`: sorted list of columns other than `target` with at most one distinct
  non-missing value.
- `unlabelled_rows` (int): rows where `target` is missing (same definition of missing).
- `positive_rate` (float): share of labelled rows with `target == 1`; 0.0 if no row is labelled.
- `bad_timestamps` (int): non-missing values in `time_col` that do not parse as dates.
- `time_range`: `[min, max]` of the parsed timestamps as ISO strings, or `[None, None]`.
- `leakage_suspects`: sorted list of columns (excluding `target`, `id_col`, `time_col`) that look
  like leaks. Compute only on labelled rows, and only when there are at least 20 labelled rows
  and both classes are present. A column is a suspect if
  - (a) it is numeric or boolean, has at least 20 non-missing labelled values, and its ROC AUC
    against the label on those rows is ≥ 0.98 or ≤ 0.02; or
  - (b) its missingness indicator (1 = missing) has ROC AUC ≥ 0.98 or ≤ 0.02 against the label
    on the labelled rows. This applies to every column, numeric or not.

  A column with a single distinct value is never a suspect.

**Clarifications** (these remove ambiguity; the grader follows them):

- *Timestamps.* In the grader's data, the valid values of a time column all share one ISO-like
  format, as in the sample (`2026-09-01 10:00`). `pd.to_datetime(..., errors="coerce")` is a
  fine way to parse. Impossible dates such as `2026-02-30 10:00` count as bad. A missing or blank
  value is not a bad timestamp.
- *`time_range`* strings may be naive (`2026-09-01T10:00:00`) or UTC (`2026-09-01T10:00:00+00:00`);
  anything `pd.Timestamp(...)` reads back as the right instant is accepted.
- *Leak check (a)* needs both classes among the column's non-missing labelled values. If they all
  come from one class, ROC AUC is undefined: (a) does not apply (do not crash), but (b) still can.
- *"A column with a single distinct value is never a suspect"*: here **missing counts as a
  value**. The rule covers only columns that are the same in every row, such as `currency`
  (always `EGP`) or a column that is entirely blank; for those, both checks are skipped. A column
  like the session's `case_closed_at`, which holds one date for every positive and is blank
  otherwise, has two "values" (the date and missing), so the rule does **not** protect it: check
  (b) applies and it is a suspect. (This is different from `constant_columns`, which ignores
  missing values.)

### `rawaj/metrics.py`

```python
def confusion_at(y_true, scores, threshold: float) -> dict: ...          # {"tp", "fp", "tn", "fn"}
def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]: ...
def precision_at_k(y_true, scores, k: int) -> float: ...
def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float: ...
def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float: ...
def threshold_for_precision(y_true, scores, min_precision: float) -> float | None: ...
```

Rules that apply to **all** of them:

- Inputs are lists or 1-D numpy arrays. Raise `ValueError` if lengths differ, the input is empty,
  a label is not 0 or 1, or a score is NaN.
- A case is flagged when `score >= threshold`.
- Precision is 0.0 when nothing is flagged; recall is 0.0 when there are no positives.
- `precision_at_k`: `ValueError` if `k <= 0`; if `k > n`, use `n`. Rank by score descending;
  ties keep their original order.
- `best_threshold`: candidates are every distinct score plus `math.inf` ("flag nothing"). Return
  the candidate with the lowest expected cost; on a tie, the **highest** threshold (fewer flags
  for the same cost). Careful: the snippet on the session slides uses ascending `np.unique` and
  `argmin`, which keeps the lowest threshold on a tie. Adapt it.
- `threshold_for_precision`: among the distinct scores, return the **lowest** threshold whose
  precision is ≥ `min_precision` (the most recall you can get while keeping the precision
  promise). Precision is not monotonic in the threshold, so check every one. Return `None` if
  no threshold reaches it.

Your functions must not read files or use the network.

### `FRAMING.md`

Four level-2 headings, exactly: `## Decision`, `## Metric`, `## Baseline`,
`## Classical ML or LLM`. Under each, a few sentences about Rawaj's mule detection. The grader
checks only that the headings exist with some text under each (text inside `<!-- -->` comments
does not count); your Track Mentor reads it.

### `AI_USAGE.md`

Short is fine. Fill in the template: the tools you used and for which functions; one place
where the AI's answer was wrong, incomplete or not what the spec said, and how you noticed (a
test, the docs, reasoning); which function you wrote without AI, and why. The grader checks
that the file exists, is not the untouched template, and has some text of your own outside
headings and comments.

## The sample data

`data/sample_transfers.csv` has 2,000 synthetic transfers with two planted leaks, a duplicate
block, blank channels, a constant column, unlabelled rows and a few broken timestamps. Run your
audit on it and see whether you find them all:

```python
import pandas as pd
from rawaj.audit import audit
df = pd.read_csv("data/sample_transfers.csv")
print(audit(df, target="is_mule", id_col="txn_id", time_col="sent_at"))
```

## How grading works

Push your work, then press **Submit for grading** on the task page on DSC Hex. The grader
clones the commit it sees at that moment, runs DSC's hidden tests on Python 3.12 with the
versions above, with the network off, 60 seconds per test and 5 minutes for the whole run, and
returns a score from 0 to 100 with feedback per test name. A test is all or nothing. Up to five
grader runs per task per day; your best attempt counts.

The hidden tests probe the edge cases this README states but the public tests do not cover,
and include property tests on random inputs, so hard-coded answers do not pass.

| Test name | What it checks | Points |
|---|---|---|
| `audit_counts_rows_and_duplicates` | Row count, duplicate IDs (sorted, missing IDs ignored), duplicate rows | 8 |
| `audit_blank_strings_are_missing` | `""`, whitespace, `None` and NaN all count as missing; numeric columns unaffected | 7 |
| `audit_constant_columns_ignore_missing` | One value plus blanks is constant; the target is never listed | 5 |
| `audit_unlabelled_rows` | Missing labels counted; positive rate on labelled rows only; all-unlabelled gives 0.0 | 7 |
| `audit_bad_timestamps_and_range` | Unparseable values counted, missing ones not; ISO range; all-bad gives `[None, None]` | 5 |
| `audit_flags_value_leak` | A numeric or boolean near-copy of the label, and its inverse, are flagged | 10 |
| `audit_flags_missingness_leak` | A column filled only for positives (or blank only for positives) is flagged | 8 |
| `audit_no_false_alarms_on_honest_features` | Honest features never flagged; the 20-row and 20-value minimums; one class flags nothing | 6 |
| `confusion_ties_at_threshold` | Scores equal to the threshold are flagged; above the max flags nothing | 7 |
| `precision_recall_zero_division` | No flags, no positives, all positives | 5 |
| `precision_at_k_ties_stable_and_k_gt_n` | Tied scores keep input order; `k > n`; `k <= 0` raises | 7 |
| `best_threshold_matches_brute_force` | Random cases against a brute-force search, including the tie rule and `inf` | 10 |
| `threshold_for_precision_lowest_valid_or_none` | Non-monotonic precision; unreachable target returns `None` | 7 |
| `input_validation` | Every function raises `ValueError` on bad input | 4 |
| `framing_and_ai_usage_present` | `FRAMING.md` headings with text; `AI_USAGE.md` filled in | 4 |

Late policy: ×1.0 within 72 hours of the session starting, ×0.6 in the following 7 days, ×0.5
after that until Thu 3 Dec 23:59. During W5 (7–13 Nov) you can resubmit at ×1.0.

AI tools are allowed; say how you used them in `AI_USAGE.md`. If you are stuck for more than
an hour, ask in the AI track channel on Discord or ask Hexa rather than spending grader runs.
