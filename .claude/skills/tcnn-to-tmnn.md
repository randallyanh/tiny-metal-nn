---
description: Migrate Python code from tinycudann (CUDA) to tiny-metal-nn (Apple Metal). Reads `tools/migrate_rules.py` for the rule tables, invokes `tools/migrate_tcnn.py` CLI for the mechanical pieces, and handles the edge cases the CLI cannot — custom losses, custom encodings, multi-file projects, and judgment calls about what to substitute when tmnn lacks a direct equivalent.
---

# tcnn → tmnn migration skill

You are migrating Python code from `tinycudann` (CUDA-only) to
`tiny_metal_nn` (Apple Metal). The CLI and rule tables already encode the
mechanical translation; your job is the **judgment** that the CLI cannot
make on its own.

## Discipline

1. **Use the existing rules. Do not invent new ones.** The single source
   of truth is `tools/migrate_rules.py`. Every alias and every reject
   came from `docs/know-how/011-json-schema-frozen.md` § 7. If the user
   asks for a translation that requires a rule not in that file, *flag
   it* — propose adding the rule to 011 + `migrate_rules.py` rather than
   doing a one-off fix that drifts the contract.

2. **Cite the rule when you apply one.** When you rewrite
   `MultiresolutionHashGrid` → `HashGrid`, say "per `migrate_rules.py`
   `ALIAS_RULES` (011 § 7.2)". This trains the user on where the
   contract lives.

3. **Don't silently substitute.** When tmnn has no equivalent (custom
   loss, fp16 path, MPS device target, etc.), surface it as a question
   *to the user* with the trade-offs spelled out — never make a quiet
   substitution.

4. **Don't run training or inference.** This skill is a translation
   tool. The user runs the migrated code on their own.

## Workflow

When the user invokes this skill (typically with one or more Python
files they want migrated):

### Step 1 — Read the rule tables

Open `tools/migrate_rules.py` and identify:
- `IDENTICAL_FIELDS` — fields that pass through unchanged.
- `ALIAS_RULES` — value rewrites with INFO/WARNING severity.
- `REJECT_RULES` — values/fields that have no tmnn equivalent.
- `_CANONICAL_OTYPES` — the otype values tmnn currently accepts.

Skim them so you know what the CLI will and won't handle.

### Step 2 — Run the CLI in `--check` mode first

For each user-provided file, run:

```
.venv/bin/python tools/migrate_tcnn.py <file> --check
```

The CLI's diagnostics tell you which mechanical pieces are already
covered (imports, simple call detection) and which need your judgment
(WARNING-level diagnostics).

### Step 3 — Handle each WARNING with judgment

The CLI flags things it cannot rewrite mechanically. For each, decide:

| CLI WARNING | Your judgment task |
|---|---|
| `tcnn.NetworkWithInputEncoding(...)` call detected | Read the surrounding code: extract the encoding config, network config, optimizer config (`torch.optim.Adam(model.parameters(), lr=...)`), and any custom learning rate / batch size, then construct the equivalent `tmnn.Trainer.from_config({...})` call. |
| Training loop with `.backward()` + `.step()` | Inspect the loss expression. If it's MSE / L1 / Huber, replace the 5-line pattern with `loss = trainer.training_step(input, target)`. If it's a custom expression, see "Custom losses" below. |
| `from tinycudann import X` | Replace with `import tiny_metal_nn as tmnn` and rewrite each downstream `X(...)` call to the `tmnn.Trainer.from_config(...)` shape. |

### Step 4 — Run the CLI again to apply mechanical pieces

```
.venv/bin/python tools/migrate_tcnn.py <file> --output <file_tmnn.py>
```

Then *layer your judgment on top of* the CLI output. Do **not** fight
the CLI by re-rewriting what it already handled.

### Step 5 — Write a final summary for the user

Show the user:
- A unified diff of the changes (mechanical + your judgment combined).
- One bullet per change explaining the rule that applied.
- A clear list of decisions you made on their behalf, each phrased as a
  question they can override (e.g., "I substituted `L2` for the custom
  MSE expression on line 42 because it's mathematically identical —
  override this if you wanted to keep manual gradient control").

---

## Custom losses

The fused `tmnn.Trainer.training_step(input, target)` only supports
built-in losses (`L2`, `L1`, `Huber`, `Cosine`). If the user's loss is
custom, you have three honest options to surface:

**Option A — substitute the closest built-in**, when the custom
expression is mathematically equivalent:

| User's expression | Substitute |
|---|---|
| `((output - target) ** 2).mean()` | `{"otype": "L2"}` (identical) |
| `(output - target).abs().mean()` | `{"otype": "L1"}` (identical) |
| `F.mse_loss(output, target)` | `{"otype": "L2"}` |
| `F.l1_loss(output, target)` | `{"otype": "L1"}` |
| `F.huber_loss(output, target, delta=δ)` | `{"otype": "Huber", "huber_delta": δ}` |
| `F.smooth_l1_loss(output, target)` | `{"otype": "Huber", "huber_delta": 1.0}` (SmoothL1 is Huber with δ=1) |

Cite that this is mathematically equivalent so the user can verify.

**Option B — keep the custom loss out of the fused path.** Use
`trainer.forward(input)` to get the model output, compute the loss in
PyTorch, call `loss.backward()` against the model's exposed parameters,
then `optimizer.step()`. This costs the autograd-boundary tax (~10-20%
overhead per 006 v2 § 5.1) but preserves arbitrary loss expressions.

> Status note: the split-path (`forward` + manual backward) Python
> binding is **not** in v1.0. As of writing, `Trainer.training_step` is
> the only training entry point exposed to Python. If the user must
> keep a custom loss, this option becomes "wait for the v1.x+ split-path
> binding" rather than "use it today". Be honest about this.

**Option C — keep the entire model in PyTorch.** If the loss is a
research-paper specialty (perceptual, contrastive, GAN-style),
migration to tmnn is probably not the right move. Tell the user that.

Pick based on the user's actual code, not a default.

---

## Custom encodings

If the user's `encoding_config.otype` is not in
`_CANONICAL_OTYPES["encoding"]` (i.e., not `HashGrid` or `RotatedMHE`)
and not in the alias map:

1. **Check `REJECT_RULES`** for a specific suggestion. e.g.,
   `SphericalHarmonics` has `suggestion="…consider HashGrid or keep SH
   in PyTorch and feed encoded features in"`.

2. **Propose the keep-in-PyTorch path** as the conservative default —
   compute the encoded features upstream of tmnn, pass them as the
   `n_input_dims` of `tmnn.Trainer.from_config(...)`. This preserves
   numerical fidelity without forcing tmnn to grow new encoders.

3. **Do not** invent a hand-rolled tmnn version of an encoder that
   doesn't exist. Stick to the supported list.

---

## Device / runtime hints

- `.to('cuda')` / `'cuda:0'` → rewrite to `.to('mps')` (or remove —
  `tmnn.Trainer` does not take a device argument since it implicitly
  uses the Metal device). Add a one-line WARNING in your summary that
  the user should re-validate numerics on Apple Silicon before relying
  on the migrated config — RNG and FP rounding can differ slightly from
  CUDA reports.

- `device='mps'` torch tensors fed into `trainer.training_step(...)` →
  the v1.0 binding rejects them with an actionable hint (call `.cpu()`
  first). Either change the user's code to call `.cpu()` upstream, or
  if the user is performance-sensitive, surface the v1.x+ MPS borrowed-
  input roadmap item (006 v2 § 12) as the proper solution and leave the
  code as-is for now.

- `torch.cuda.amp` / `loss_scale` → tmnn v1.0 is fp32-only. Remove the
  amp setup; flag the expected ~50% performance ceiling vs fp16 in your
  summary so the user is calibrated.

- Multi-GPU / DDP scaffolding → Apple Silicon is single-GPU. Strip the
  init code and warn that any `world_size`-aware logic needs to collapse
  to single-GPU before this script will run.

---

## Multi-file projects

The CLI processes one file at a time and cannot trace a `model =
build_model(...)` factory across modules. When the migration spans
multiple files:

1. Identify the file that defines the tcnn entry point (the
   `tcnn.NetworkWithInputEncoding(...)` call). Run the CLI on it first.
2. Find downstream files that import or depend on that entry point.
   Update their imports + any direct `tcnn.X` references manually.
3. If the user's project uses a config registry (e.g., a YAML loaded
   into a tcnn config dict), translate the registry contents using the
   same `migrate_rules.translate_config` data — preferably by importing
   it and calling `translate_config` on the loaded dict, then writing
   the result back.

---

## Output format

Always end with:

1. **Unified diff** of all changes (use `difflib.unified_diff` or
   `git diff`-style output).
2. **One-bullet-per-change rationale** with rule citations.
3. **"Decisions I made — override these if needed"** section, listing
   judgment calls (loss substitutions, device changes, dropped fp16,
   etc.).
4. **"Cannot run yet" checklist** if the migrated file still has
   unresolved warnings — give the user the exact line numbers and a
   minimal patch they can paste.

---

## Anti-patterns (do not do these)

- ❌ Adding a new alias / reject rule directly in chat without first
  proposing the addition to `migrate_rules.py` + 011 § 7. The contract
  must stay single-sourced.
- ❌ Running the migrated code to "validate" it. tmnn requires Metal
  hardware, build artifacts, and the user's data — none of which the
  skill should assume.
- ❌ Silent device redirects (`'cuda'` → `'mps'` without flagging the
  numerics caveat).
- ❌ Promising fp16 will work post-migration. It will not in v1.0.
- ❌ Writing a long English explanation when the user's code can be
  pointed to directly. Cite line numbers and rules.

---

## Reference docs (in priority order)

- `docs/know-how/011-json-schema-frozen.md` § 7 — the JSON schema diff
  source of truth.
- `docs/know-how/006-python-binding-design.md` v2 § 4 / § 10 — the
  migration tooling design contract (CLI, skill, examples three-piece).
- `docs/TCNN-MIGRATION-GUIDE.md` § 10 — user-facing summary of the
  three migration paths and quick-reference field table.
- `tools/migrate_rules.py` — the executable rule tables.
- `tools/migrate_tcnn.py` — the CLI you delegate the mechanical pieces
  to.
