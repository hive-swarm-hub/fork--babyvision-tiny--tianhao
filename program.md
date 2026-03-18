# BabyVision Tiny Solver

Improve a visual reasoning solver on BabyVision (30-problem subset).

## Setup

1. **Read the in-scope files**:
   - `agent.py` — the file you modify. The visual reasoning solver.
   - `eval/eval.sh` — runs evaluation. Do not modify.
   - `prepare.sh` — downloads and samples the dataset + images. Do not modify.
2. **Run prepare**: `bash prepare.sh` to download the dataset. Saves images to `data/images/`.
3. **Verify data exists**: Check that `data/` contains `test.jsonl` and `data/images/` has .jpg files.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row.
5. **Run baseline**: `bash eval/eval.sh` to establish the starting accuracy.

## The benchmark

BabyVision Tiny is a 30-problem subset (seed=42) of BabyVision, testing fine-grained visual perception across task subtypes:
- **Pattern matching**: Find identical/different patterns in grids
- **Counting**: Count specific shapes, clusters, 3D blocks
- **Spatial**: Mazes, line connections, shadow matching
- **Recognition**: Numbers, letters, 3D views, cube unfolding
- **Transformation**: Rotation, mirroring, paper folding

The agent receives images as base64 via the OpenAI vision API.

**Note**: Requires a **vision model** (e.g., gpt-5.4-mini, gpt-4o). Set `SOLVER_MODEL` accordingly.

## Answer format

- **Choice questions** (`ans_type=choice`): Answer with the option number (1, 2, 3, or 4).
- **Blank questions** (`ans_type=blank`): Answer with the exact value — coordinates like `(4,7)`, counts like `14`, etc.

## Experimentation

**What you CAN do:**
- Modify `agent.py` — this is the only file you edit. Everything is fair game: prompting strategy, image analysis, chain-of-thought, describing the image before answering, zooming strategies.

**What you CANNOT do:**
- Modify `eval/`, `prepare.sh`, or test data.
- Change the model (set via `SOLVER_MODEL` env var).
- Install new packages beyond what's in `requirements.txt`.

**The goal: maximize accuracy.** Choice = exact option number. Blank = exact string match (case-insensitive). Accuracy = fraction correct.

**Cost** is a soft constraint.

**Simplicity criterion**: All else being equal, simpler is better.

## Output format

```
---
accuracy:         0.2000
correct:          6
total:            30
```

## Logging results

Log each experiment to `results.tsv` (tab-separated):

```
commit	accuracy	cost_usd	status	description
a1b2c3d	0.200000	2.00	keep	baseline
b2c3d4e	0.330000	3.50	keep	describe-then-answer + grid analysis
```

## The experiment loop

LOOP FOREVER:

1. **THINK** — decide what to try next. Review results.tsv. These tasks test fine-grained visual perception — consider having the model describe the image in detail before answering, or breaking the image into regions.
2. Modify `agent.py` with your experimental idea.
3. git commit
4. Run the experiment: `bash eval/eval.sh > run.log 2>&1`
5. Read out the results: `grep "^accuracy:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` for the stack trace and attempt a fix.
7. Record the results in results.tsv (do not commit results.tsv).
8. If accuracy improved (higher), keep the git commit. If equal or worse, `git reset --hard HEAD~1`.

**Timeout**: If a run exceeds 30 minutes, kill it and treat it as a failure.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. The loop runs until interrupted.
