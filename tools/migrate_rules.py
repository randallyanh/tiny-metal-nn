"""tcnn → tmnn migration translation rules — single source of truth.

This file is consumed by all three migration deliverables (006 v2 §1.4):

  - tools/migrate_tcnn.py           CLI translator      (stage 8)
  - .claude/skills/tcnn-to-tmnn.md  Claude Code skill   (stage 9)
  - docs/TCNN-MIGRATION-GUIDE.md    user-facing summary (already published)

Schema contract: docs/know-how/011-json-schema-frozen.md (the freeze doc).
Every rule in this file MUST correspond to an entry in 011's §7 tables.
The self-test at the bottom enforces internal consistency; CI will eventually
add a cross-check against 011's tables once the schema is finalized.

Scope of this file:
  - JSON config-dict translation rules (alias / rewrite / reject)
  - Translation-direction metadata (which fields stay identical, which warn)

Out of scope (lives elsewhere):
  - AST rewriting of training loops, imports, .to('cuda')   → tools/migrate_tcnn.py
  - File I/O, CLI argument parsing, --diff rendering        → tools/migrate_tcnn.py
  - Project-wide refactor heuristics                        → Claude Code skill

Version: 0.1 (stage-1 skeleton; translation function is intentionally stubbed)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Diagnostic types
# ---------------------------------------------------------------------------

class Severity(Enum):
    """Diagnostic severity. Mirrors C++ tmnn::DiagnosticSeverity."""
    INFO = "info"        # alias normalization; user does not have to act
    WARNING = "warning"  # rewrite with semantic shift; user should review
    ERROR = "error"      # reject; user must fix manually


@dataclass(frozen=True)
class Diagnostic:
    severity: Severity
    path: str            # JSON path, e.g., "encoding.otype"
    message: str         # human-readable explanation
    suggestion: Optional[str] = None  # actionable next step (mostly for ERROR)


# ---------------------------------------------------------------------------
# Rule types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AliasRule:
    """Rewrite a tcnn value to the canonical tmnn value.

    INFO severity = pure name alias (semantics identical).
    WARNING severity = no exact tmnn equivalent; the closest match is used and
                       the user should be aware of any semantic shift.
    """
    path: str            # e.g., "encoding.otype"
    from_value: str      # tcnn value as written
    to_value: str        # tmnn canonical value
    severity: Severity   # INFO or WARNING
    message: str


@dataclass(frozen=True)
class RejectRule:
    """Reject a value or field outright (the user must intervene).

    `value=None` rejects the field's *presence* regardless of value
    (e.g., `optimizer.relative_decay` has no Adam equivalent).
    """
    path: str
    value: Optional[str]   # None = reject any value of this field
    message: str           # why we cannot translate
    suggestion: str        # what the user should do instead
    in_v1x_roadmap: bool   # True ⇒ tmnn plans to support this in v1.x+


# ---------------------------------------------------------------------------
# Translation tables — match docs/know-how/011-json-schema-frozen.md §7
# ---------------------------------------------------------------------------

# §7.1 — fields whose name + semantics are tcnn-tmnn identical (no rewrite,
# no diagnostic). Grouped by top-level section for readability.
IDENTICAL_FIELDS: Dict[str, List[str]] = {
    "encoding": [
        "n_levels",
        "n_features_per_level",
        "log2_hashmap_size",
        "base_resolution",
        "per_level_scale",
        # Field name passes through; value-level rejects (Smoothstep / Nearest)
        # are handled by REJECT_RULES.
        "interpolation",
    ],
    "network": [
        "n_neurons",
        "n_hidden_layers",
        # Field names pass through; value-level rejects (non-ReLU activations,
        # non-None output activations) are handled by REJECT_RULES, and the
        # `Linear` → `None` rewrite is handled by ALIAS_RULES.
        "activation",
        "output_activation",
    ],
    "loss": [
        "huber_delta",
        "output_dims",  # Cosine loss only
    ],
    "optimizer": [
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "l1_reg",
        "l2_reg",
    ],
}


# §7.2 — alias / rewrite rules
ALIAS_RULES: List[AliasRule] = [
    # encoding
    AliasRule(
        path="encoding.otype",
        from_value="MultiresolutionHashGrid",
        to_value="HashGrid",
        severity=Severity.INFO,
        message="encoding.otype 'MultiresolutionHashGrid' is the legacy name; tmnn canonical is 'HashGrid'.",
    ),
    AliasRule(
        path="encoding.otype",
        from_value="RotatedMultiresHashGrid",
        to_value="RotatedMHE",
        severity=Severity.INFO,
        message="encoding.otype 'RotatedMultiresHashGrid' is the legacy name; tmnn canonical is 'RotatedMHE'.",
    ),
    # network
    AliasRule(
        path="network.output_activation",
        from_value="Linear",
        to_value="None",
        severity=Severity.INFO,
        message="network.output_activation 'Linear' is normalized to 'None' in tmnn (identity output).",
    ),
    AliasRule(
        path="network.otype",
        from_value="CutlassMLP",
        to_value="FullyFusedMLP",
        severity=Severity.WARNING,
        message=(
            "network.otype 'CutlassMLP' has no Apple Metal equivalent; using 'FullyFusedMLP'. "
            "Throughput characteristics may differ from the original tcnn report."
        ),
    ),
]


# §7.3 — values that tmnn v1.0 rejects (user must intervene)
REJECT_RULES: List[RejectRule] = [
    # encoding
    RejectRule(
        path="encoding.otype",
        value="Frequency",
        message="Frequency encoding is not implemented in tmnn v1.0.",
        suggestion="Use 'HashGrid' as the closest tmnn equivalent, or keep the Frequency encoding in PyTorch and feed encoded features into tmnn.",
        in_v1x_roadmap=True,
    ),
    RejectRule(
        path="encoding.otype",
        value="OneBlob",
        message="OneBlob encoding is not implemented in tmnn v1.0.",
        suggestion="Use 'HashGrid' as the closest tmnn equivalent.",
        in_v1x_roadmap=True,
    ),
    RejectRule(
        path="encoding.otype",
        value="SphericalHarmonics",
        message="Spherical harmonics encoding is out of scope for tmnn (regression-focused).",
        suggestion="Either keep SH in PyTorch and feed encoded features into tmnn, or substitute 'HashGrid'.",
        in_v1x_roadmap=False,
    ),
    RejectRule(
        path="encoding.otype",
        value="Identity",
        message="Identity encoding is unnecessary in tmnn.",
        suggestion="Remove the encoding wrapper and pass raw input directly to the network.",
        in_v1x_roadmap=False,
    ),
    RejectRule(
        path="encoding.interpolation",
        value="Smoothstep",
        message="HashGrid interpolation 'Smoothstep' is not implemented in tmnn v1.0.",
        suggestion="Use 'Linear' (the v1.0 default).",
        in_v1x_roadmap=True,
    ),
    RejectRule(
        path="encoding.interpolation",
        value="Nearest",
        message="HashGrid interpolation 'Nearest' is not implemented in tmnn v1.0.",
        suggestion="Use 'Linear' (the v1.0 default).",
        in_v1x_roadmap=True,
    ),
    # network
    RejectRule(
        path="network.otype",
        value="TiledMLP",
        message="TiledMLP is reserved for tmnn C4; not yet wired to factory_json.",
        suggestion="Use 'FullyFusedMLP' for v1.0.",
        in_v1x_roadmap=True,
    ),
    # network.activation — v1.0 schema accepts ReLU only (per 011 §0.2)
    RejectRule(
        path="network.activation",
        value="Sigmoid",
        message="network.activation 'Sigmoid' is not accepted by the v1.0 schema (FullyFusedMLP runtime is ReLU-only today).",
        suggestion="Either substitute 'ReLU' or wait for v1.x+ runtime extension.",
        in_v1x_roadmap=True,
    ),
    RejectRule(
        path="network.activation",
        value="Tanh",
        message="network.activation 'Tanh' is not accepted by the v1.0 schema.",
        suggestion="Either substitute 'ReLU' or wait for v1.x+ runtime extension.",
        in_v1x_roadmap=True,
    ),
    RejectRule(
        path="network.activation",
        value="LeakyReLU",
        message="network.activation 'LeakyReLU' is not accepted by the v1.0 schema.",
        suggestion="Either substitute 'ReLU' or wait for v1.x+ runtime extension.",
        in_v1x_roadmap=True,
    ),
    RejectRule(
        path="network.activation",
        value="Linear",
        message="network.activation 'Linear' is not accepted by the v1.0 schema (use output_activation for the final-layer linearity).",
        suggestion="Substitute 'ReLU' for hidden layers; 'Linear' on output_activation is accepted.",
        in_v1x_roadmap=True,
    ),
    RejectRule(
        path="network.activation",
        value="Squareplus",
        message="network.activation 'Squareplus' is out of scope for tmnn.",
        suggestion="Substitute 'ReLU'.",
        in_v1x_roadmap=False,
    ),
    RejectRule(
        path="network.activation",
        value="Softplus",
        message="network.activation 'Softplus' is out of scope for tmnn.",
        suggestion="Substitute 'ReLU'.",
        in_v1x_roadmap=False,
    ),
    RejectRule(
        path="network.activation",
        value="Snake",
        message="network.activation 'Snake' (SIREN-style) is out of scope for tmnn.",
        suggestion="Use Uniform weight init with a tuned bound and ReLU, or keep this network in PyTorch.",
        in_v1x_roadmap=False,
    ),
    RejectRule(
        path="network.activation",
        value="Sine",
        message="network.activation 'Sine' (SIREN) is out of scope for tmnn.",
        suggestion="Keep SIREN in PyTorch; use tmnn for downstream hash-grid + ReLU stages.",
        in_v1x_roadmap=False,
    ),
    # loss
    RejectRule(
        path="loss.otype",
        value="RelativeL2",
        message="RelativeL2 loss is not implemented in tmnn v1.0.",
        suggestion="Substitute 'L2' (relative-vs-absolute typically affects scaling, not optimum).",
        in_v1x_roadmap=True,
    ),
    RejectRule(
        path="loss.otype",
        value="SmoothL1",
        message="SmoothL1 loss is not implemented in tmnn v1.0.",
        suggestion="Substitute 'Huber' (SmoothL1 is Huber with delta=1.0).",
        in_v1x_roadmap=True,
    ),
    RejectRule(
        path="loss.otype",
        value="CrossEntropy",
        message="CrossEntropy loss is out of scope for tmnn (regression-focused).",
        suggestion="Keep classification training in PyTorch.",
        in_v1x_roadmap=False,
    ),
    # optimizer
    RejectRule(
        path="optimizer.otype",
        value="Shampoo",
        message="Shampoo optimizer is out of scope for tmnn v1.0.",
        suggestion="Substitute 'Adam'.",
        in_v1x_roadmap=False,
    ),
    RejectRule(
        path="optimizer.otype",
        value="Novograd",
        message="Novograd optimizer is out of scope for tmnn v1.0.",
        suggestion="Substitute 'Adam'.",
        in_v1x_roadmap=False,
    ),
    RejectRule(
        path="optimizer.otype",
        value="SGD",
        message="SGD optimizer is not implemented in tmnn (Adam is the default and only option).",
        suggestion="Substitute 'Adam'; for SGD-like behavior set high beta1 / beta2.",
        in_v1x_roadmap=False,
    ),
    # field-level rejections (presence-based, value-agnostic)
    RejectRule(
        path="optimizer.relative_decay",
        value=None,
        message="Adam in tmnn does not have 'relative_decay'; the field is unknown.",
        suggestion="Remove the field; use 'l1_reg' / 'l2_reg' for regularization.",
        in_v1x_roadmap=False,
    ),
    RejectRule(
        path="loss_scale",  # top-level
        value=None,
        message="loss_scale is tied to fp16 mixed precision; tmnn v1.0 is fp32-only.",
        suggestion="Remove the field. fp16 support is on the v1.x+ roadmap.",
        in_v1x_roadmap=True,
    ),
]


# §7.4 — fields that exist in tmnn but not in tcnn. The CLI knows about these
# so that re-running on a partially-migrated file does not flag them as
# "unknown". Listed as canonical paths (or path:value for otype-specific).
TMNN_ONLY_FIELDS: List[str] = [
    "encoding.otype:RotatedMHE",
    "weight_init",
    "weight_init.hash_grid_init",
    "weight_init.hash_grid_range",
    "weight_init.mlp_init",
    "weight_init.mlp_nonlinearity",
    "weight_init.mlp_uniform_range",
    "weight_init.mlp_normal_stddev",
    "weight_init.mlp_kaiming_a",
    "weight_init.seed",
    "batch_size",
]


# Default sub-configs that tmnn factory_json substitutes when missing
# (mirror C++ behavior so the CLI can render the migrated config explicitly
# rather than relying on factory_json substitution silently).
DEFAULT_LOSS = {"otype": "L2"}
DEFAULT_OPTIMIZER = {"otype": "Adam"}


# Canonical otype values accepted by tmnn factory_json today (011 §2-§5).
# Anything not in here that isn't an alias (ALIAS_RULES) or a known reject
# (REJECT_RULES) is an unrecognized otype.
_CANONICAL_OTYPES: Dict[str, frozenset[str]] = {
    "encoding": frozenset({"HashGrid", "RotatedMHE"}),
    "network": frozenset({"FullyFusedMLP"}),
    "loss": frozenset({"L2", "L1", "Huber", "Cosine"}),
    "optimizer": frozenset({"Adam"}),
}

# Top-level keys that route to a section translator.
_TOP_LEVEL_SECTIONS: frozenset[str] = frozenset(
    {"encoding", "network", "loss", "optimizer"}
)

# tmnn-native top-level keys that may already be present when the user runs
# the migrator on a partially-migrated file. Pass them through unchanged.
_TMNN_ONLY_TOP_LEVEL: frozenset[str] = frozenset(
    {"weight_init", "batch_size", "training"}
)


# ---------------------------------------------------------------------------
# Translation API (stub for stage 1; implemented in stage 8)
# ---------------------------------------------------------------------------

def translate_config(
    tcnn_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Diagnostic]]:
    """Translate a tcnn-shape JSON config dict to tmnn-shape.

    Returns:
        (tmnn_config, diagnostics)
        tmnn_config: dict suitable for tmnn.Trainer.from_config(...)
        diagnostics: list of Diagnostic; any Severity.ERROR means the user
                     must intervene (the returned tmnn_config is incomplete).
    """
    diagnostics: List[Diagnostic] = []

    if not isinstance(tcnn_config, dict):
        diagnostics.append(
            Diagnostic(
                severity=Severity.ERROR,
                path="config",
                message=(
                    f"top-level config must be a dict; got "
                    f"{type(tcnn_config).__name__}"
                ),
            )
        )
        return {}, diagnostics

    out: Dict[str, Any] = {}

    for top_key, value in tcnn_config.items():
        if top_key in _TOP_LEVEL_SECTIONS:
            translated, sub_diag = _translate_section(top_key, value)
            out[top_key] = translated
            diagnostics.extend(sub_diag)
        elif top_key in _TMNN_ONLY_TOP_LEVEL:
            out[top_key] = value
            if top_key == "training":
                diagnostics.append(
                    Diagnostic(
                        severity=Severity.INFO,
                        path="training",
                        message=(
                            "field is parsed but ignored by tmnn factory_json "
                            "(passed through to keep the config round-trippable)"
                        ),
                    )
                )
        else:
            # Check whether this top-level field is on the explicit reject list.
            reject = _find_reject(path=top_key, value=None)
            if reject:
                diagnostics.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        path=top_key,
                        message=reject.message,
                        suggestion=reject.suggestion,
                    )
                )
            else:
                diagnostics.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        path=top_key,
                        message=(
                            f"unknown top-level field '{top_key}'. "
                            "tmnn accepts: encoding, network, loss, optimizer, "
                            "weight_init, batch_size, training."
                        ),
                    )
                )

    # Default substitution mirrors what tmnn::factory_json would do at the
    # C++ layer. Emitting it explicitly in the migrated config makes the
    # output self-documenting.
    if "loss" not in out:
        out["loss"] = dict(DEFAULT_LOSS)
        diagnostics.append(
            Diagnostic(
                severity=Severity.INFO,
                path="loss",
                message="default L2 loss substituted (no loss section in tcnn config)",
            )
        )
    if "optimizer" not in out:
        out["optimizer"] = dict(DEFAULT_OPTIMIZER)
        diagnostics.append(
            Diagnostic(
                severity=Severity.INFO,
                path="optimizer",
                message="default Adam optimizer substituted (no optimizer section in tcnn config)",
            )
        )

    return out, diagnostics


def _translate_section(
    section: str, sub_config: Any
) -> Tuple[Dict[str, Any], List[Diagnostic]]:
    """Translate one section (encoding / network / loss / optimizer)."""
    diagnostics: List[Diagnostic] = []

    if not isinstance(sub_config, dict):
        diagnostics.append(
            Diagnostic(
                severity=Severity.ERROR,
                path=section,
                message=(
                    f"section must be a dict; got {type(sub_config).__name__}"
                ),
            )
        )
        return {}, diagnostics

    out: Dict[str, Any] = {}
    identical_fields = set(IDENTICAL_FIELDS.get(section, []))

    for field, value in sub_config.items():
        full_path = f"{section}.{field}"

        # 1. Alias rules — rewrite known legacy / equivalent values.
        alias = _find_alias(path=full_path, from_value=str(value))
        if alias is not None:
            out[field] = alias.to_value
            diagnostics.append(
                Diagnostic(
                    severity=alias.severity,
                    path=full_path,
                    message=alias.message,
                )
            )
            continue

        # 2. Reject rules — explicit value match wins, fall back to
        # value-agnostic field-level reject.
        reject = _find_reject(path=full_path, value=str(value))
        if reject is None:
            reject = _find_reject(path=full_path, value=None)
        if reject is not None:
            diagnostics.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    path=full_path,
                    message=reject.message,
                    suggestion=reject.suggestion,
                )
            )
            continue

        # 3. otype: must be a tmnn-canonical value (or already aliased above).
        if field == "otype":
            canonical = _CANONICAL_OTYPES.get(section, frozenset())
            if str(value) in canonical:
                out[field] = value
                continue
            diagnostics.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    path=full_path,
                    message=(
                        f"unknown otype '{value}' for {section}; tmnn knows: "
                        f"{sorted(canonical)}"
                    ),
                )
            )
            continue

        # 4. Identical fields — pass through unchanged.
        if field in identical_fields:
            out[field] = value
            continue

        # 5. Unknown field for this section.
        diagnostics.append(
            Diagnostic(
                severity=Severity.ERROR,
                path=full_path,
                message=(
                    f"unknown field '{field}' in {section}; supported: "
                    f"{sorted(identical_fields | {'otype'})}"
                ),
            )
        )

    return out, diagnostics


def _find_alias(*, path: str, from_value: str) -> Optional[AliasRule]:
    for r in ALIAS_RULES:
        if r.path == path and r.from_value == from_value:
            return r
    return None


def _find_reject(*, path: str, value: Optional[str]) -> Optional[RejectRule]:
    for r in REJECT_RULES:
        if r.path == path and r.value == value:
            return r
    return None


# ---------------------------------------------------------------------------
# Self-tests (run with `python tools/migrate_rules.py`)
# ---------------------------------------------------------------------------

# Schema paths that 011 freeze actually validates. Used by self-tests to make
# sure no rule references a path that doesn't exist in the schema.
_KNOWN_SCHEMA_PATHS = frozenset([
    "encoding.otype",
    "encoding.n_levels",
    "encoding.n_features_per_level",
    "encoding.log2_hashmap_size",
    "encoding.base_resolution",
    "encoding.per_level_scale",
    "encoding.interpolation",
    "network.otype",
    "network.n_neurons",
    "network.n_hidden_layers",
    "network.activation",
    "network.output_activation",
    "loss.otype",
    "loss.huber_delta",
    "loss.output_dims",
    "optimizer.otype",
    "optimizer.learning_rate",
    "optimizer.beta1",
    "optimizer.beta2",
    "optimizer.epsilon",
    "optimizer.l1_reg",
    "optimizer.l2_reg",
    # 011 §6 weight_init
    "weight_init.hash_grid_init",
    "weight_init.hash_grid_range",
    "weight_init.mlp_init",
    "weight_init.mlp_nonlinearity",
    "weight_init.mlp_uniform_range",
    "weight_init.mlp_normal_stddev",
    "weight_init.mlp_kaiming_a",
    "weight_init.seed",
    # top-level
    "batch_size",
    "training",
])

# Paths that REJECT_RULES may reference but are not in the tmnn schema (they
# only appear in tcnn config and are rejected outright on the way in).
_TCNN_ONLY_REJECT_PATHS = frozenset([
    "optimizer.relative_decay",
    "loss_scale",
])


def _validate_table_consistency() -> List[str]:
    """Sanity-check internal consistency of the rule tables.

    Returns a list of issue strings; empty list = all good.
    """
    issues: List[str] = []

    # IDENTICAL_FIELDS: every full path is a known schema path
    for section, fields in IDENTICAL_FIELDS.items():
        for f in fields:
            full = f"{section}.{f}"
            if full not in _KNOWN_SCHEMA_PATHS:
                issues.append(f"IDENTICAL_FIELDS[{section}] has '{f}' which is not a known schema path '{full}'")

    # ALIAS_RULES: every path is known; every from→to value is non-empty;
    # severity is INFO or WARNING (never ERROR).
    for r in ALIAS_RULES:
        if r.path not in _KNOWN_SCHEMA_PATHS:
            issues.append(f"AliasRule path '{r.path}' is not a known schema path")
        if not r.from_value or not r.to_value:
            issues.append(f"AliasRule {r.path}: empty from_value or to_value")
        if r.severity == Severity.ERROR:
            issues.append(f"AliasRule {r.path} ({r.from_value}→{r.to_value}): severity must be INFO or WARNING, not ERROR")

    # REJECT_RULES: path is known schema path or in _TCNN_ONLY_REJECT_PATHS.
    # suggestion is non-empty.
    for r in REJECT_RULES:
        if (r.path not in _KNOWN_SCHEMA_PATHS) and (r.path not in _TCNN_ONLY_REJECT_PATHS):
            issues.append(f"RejectRule path '{r.path}' is not in known schema paths or tcnn-only paths")
        if not r.message.strip() or not r.suggestion.strip():
            issues.append(f"RejectRule {r.path}/{r.value}: empty message or suggestion")

    # No (path, value) appears in both ALIAS_RULES and REJECT_RULES — that
    # would be contradictory.
    alias_keys = {(r.path, r.from_value) for r in ALIAS_RULES}
    reject_keys = {(r.path, r.value) for r in REJECT_RULES if r.value is not None}
    overlap = alias_keys & reject_keys
    if overlap:
        issues.append(f"Overlap between ALIAS_RULES and REJECT_RULES: {overlap}")

    return issues


def _summary() -> str:
    return (
        f"migrate_rules.py — stage-1 skeleton\n"
        f"  IDENTICAL_FIELDS: {sum(len(v) for v in IDENTICAL_FIELDS.values())} fields "
        f"across {len(IDENTICAL_FIELDS)} sections\n"
        f"  ALIAS_RULES:      {len(ALIAS_RULES)} ({sum(1 for r in ALIAS_RULES if r.severity == Severity.INFO)} info, "
        f"{sum(1 for r in ALIAS_RULES if r.severity == Severity.WARNING)} warning)\n"
        f"  REJECT_RULES:     {len(REJECT_RULES)} "
        f"({sum(1 for r in REJECT_RULES if r.in_v1x_roadmap)} on v1.x+ roadmap, "
        f"{sum(1 for r in REJECT_RULES if not r.in_v1x_roadmap)} out of scope)\n"
        f"  TMNN_ONLY_FIELDS: {len(TMNN_ONLY_FIELDS)}\n"
    )


if __name__ == "__main__":
    issues = _validate_table_consistency()
    if issues:
        print("FAIL: migration rules consistency check found issues:")
        for i in issues:
            print(f"  - {i}")
        raise SystemExit(1)
    print("OK")
    print(_summary())
