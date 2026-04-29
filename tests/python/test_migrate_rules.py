"""Tests for tools/migrate_rules.py — the translation rules consumed by
the migration CLI (`tools/migrate_tcnn.py`) and the Claude Code skill.

Pure-Python tests; no GPU / no _C binding required. Source of truth for
the mappings is `docs/know-how/011-json-schema-frozen.md` § 7 (the freeze
doc).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `tools/` importable from the test process (it lives at the repo
# root, not on the wheel install path).
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tools"))

import migrate_rules as mr  # noqa: E402


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_identical_config_passes_through_with_default_substitutions():
    """A fully-canonical tcnn config is accepted; only INFO diagnostics fire."""
    config = {
        "encoding": {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16.0,
            "per_level_scale": 1.447,
        },
        "network": {
            "otype": "FullyFusedMLP",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        },
    }
    out, diags = mr.translate_config(config)

    # Defaults injected for missing loss/optimizer.
    assert out["encoding"]["otype"] == "HashGrid"
    assert out["network"]["otype"] == "FullyFusedMLP"
    assert out["loss"] == {"otype": "L2"}
    assert out["optimizer"] == {"otype": "Adam"}

    # Only INFO-level (default substitutions) — no errors / warnings.
    assert all(d.severity == mr.Severity.INFO for d in diags)
    assert any(
        d.path == "loss" and "default L2" in d.message for d in diags
    )
    assert any(
        d.path == "optimizer" and "default Adam" in d.message for d in diags
    )


def test_alias_rewrite_emits_info_diagnostic():
    """MultiresolutionHashGrid is the legacy name; output uses HashGrid."""
    config = {
        "encoding": {"otype": "MultiresolutionHashGrid", "n_levels": 8},
        "network": {"otype": "FullyFusedMLP"},
    }
    out, diags = mr.translate_config(config)
    assert out["encoding"]["otype"] == "HashGrid"
    assert out["encoding"]["n_levels"] == 8
    info = [
        d for d in diags
        if d.path == "encoding.otype" and d.severity == mr.Severity.INFO
    ]
    assert len(info) == 1
    assert "legacy" in info[0].message.lower()


def test_cutlass_mlp_aliased_with_warning():
    """CutlassMLP has no Apple Metal equivalent → mapped to FullyFusedMLP + WARNING."""
    config = {
        "encoding": {"otype": "HashGrid"},
        "network": {"otype": "CutlassMLP", "n_neurons": 64},
    }
    out, diags = mr.translate_config(config)
    assert out["network"]["otype"] == "FullyFusedMLP"
    warns = [
        d for d in diags
        if d.path == "network.otype" and d.severity == mr.Severity.WARNING
    ]
    assert len(warns) == 1
    assert "CutlassMLP" in warns[0].message


# ---------------------------------------------------------------------------
# Reject paths
# ---------------------------------------------------------------------------


def test_unsupported_encoding_otype_emits_error_with_suggestion():
    """SphericalHarmonics is out of scope; user gets a suggestion."""
    config = {
        "encoding": {"otype": "SphericalHarmonics"},
        "network": {"otype": "FullyFusedMLP"},
    }
    out, diags = mr.translate_config(config)
    errors = [
        d for d in diags
        if d.path == "encoding.otype" and d.severity == mr.Severity.ERROR
    ]
    assert len(errors) == 1
    assert "out of scope" in errors[0].message.lower()
    assert errors[0].suggestion is not None
    assert "HashGrid" in errors[0].suggestion or "PyTorch" in errors[0].suggestion


def test_loss_scale_top_level_rejected_with_fp16_note():
    """loss_scale ties to fp16 mixed precision; tmnn v1.0 is fp32-only."""
    config = {
        "encoding": {"otype": "HashGrid"},
        "network": {"otype": "FullyFusedMLP"},
        "loss_scale": 128.0,
    }
    _, diags = mr.translate_config(config)
    errors = [d for d in diags if d.path == "loss_scale"]
    assert len(errors) == 1
    assert "fp16" in errors[0].message.lower() or "fp32" in errors[0].message.lower()


def test_unsupported_optimizer_otype_rejected():
    """Shampoo is out of scope; suggestion points to Adam."""
    config = {
        "encoding": {"otype": "HashGrid"},
        "network": {"otype": "FullyFusedMLP"},
        "optimizer": {"otype": "Shampoo"},
    }
    out, diags = mr.translate_config(config)
    errors = [d for d in diags if d.path == "optimizer.otype"]
    assert len(errors) == 1
    assert "Adam" in errors[0].suggestion


def test_unsupported_loss_otype_with_v1x_roadmap_hint():
    """RelativeL2 is on the roadmap; suggestion points at L2 substitute."""
    config = {
        "encoding": {"otype": "HashGrid"},
        "network": {"otype": "FullyFusedMLP"},
        "loss": {"otype": "RelativeL2"},
    }
    _, diags = mr.translate_config(config)
    errors = [d for d in diags if d.path == "loss.otype"]
    assert len(errors) == 1
    assert "L2" in errors[0].suggestion


def test_unknown_field_in_encoding_section_rejected():
    config = {
        "encoding": {"otype": "HashGrid", "mystery": 42},
        "network": {"otype": "FullyFusedMLP"},
    }
    _, diags = mr.translate_config(config)
    errs = [d for d in diags if d.path == "encoding.mystery"]
    assert len(errs) == 1
    assert errs[0].severity == mr.Severity.ERROR


# ---------------------------------------------------------------------------
# Pass-through for tmnn-only fields (re-running on partially-migrated config)
# ---------------------------------------------------------------------------


def test_tmnn_only_fields_pass_through():
    """If the user already has weight_init / batch_size, leave them alone."""
    config = {
        "encoding": {"otype": "HashGrid"},
        "network": {"otype": "FullyFusedMLP"},
        "weight_init": {"seed": 12345, "mlp_init": "KaimingUniform"},
        "batch_size": 4096,
    }
    out, _ = mr.translate_config(config)
    assert out["weight_init"] == {"seed": 12345, "mlp_init": "KaimingUniform"}
    assert out["batch_size"] == 4096


def test_training_top_level_passes_through_with_info():
    """tcnn often has a top-level `training` dict; tmnn ignores it but passes
    it through so the migrated config stays round-trippable."""
    config = {
        "encoding": {"otype": "HashGrid"},
        "network": {"otype": "FullyFusedMLP"},
        "training": {"n_steps": 1000, "lr_schedule": "constant"},
    }
    out, diags = mr.translate_config(config)
    assert out["training"] == {"n_steps": 1000, "lr_schedule": "constant"}
    info = [d for d in diags if d.path == "training"]
    assert len(info) == 1
    assert info[0].severity == mr.Severity.INFO


# ---------------------------------------------------------------------------
# Input shape / safety
# ---------------------------------------------------------------------------


def test_non_dict_input_emits_clear_error():
    out, diags = mr.translate_config([1, 2, 3])  # type: ignore[arg-type]
    assert out == {}
    errs = [d for d in diags if d.severity == mr.Severity.ERROR]
    assert errs and errs[0].path == "config"


def test_non_dict_section_value_emits_clear_error():
    out, diags = mr.translate_config({"encoding": "HashGrid"})
    errs = [d for d in diags if d.path == "encoding"]
    assert errs
    assert errs[0].severity == mr.Severity.ERROR


def test_huber_loss_keeps_huber_delta():
    """Field-level pass-through inside loss section."""
    config = {
        "encoding": {"otype": "HashGrid"},
        "network": {"otype": "FullyFusedMLP"},
        "loss": {"otype": "Huber", "huber_delta": 0.5},
    }
    out, diags = mr.translate_config(config)
    assert out["loss"] == {"otype": "Huber", "huber_delta": 0.5}
    # No ERROR diagnostics for this section.
    assert not [
        d for d in diags
        if d.path.startswith("loss.") and d.severity == mr.Severity.ERROR
    ]


def test_realistic_instant_ngp_config_round_trips_cleanly():
    """A typical instant-NGP-shaped tcnn config translates with only the
    default-loss / default-optimizer info diagnostics."""
    config = {
        "encoding": {
            "otype": "MultiresolutionHashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 1.5,
            "interpolation": "Linear",
        },
        "network": {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        },
    }
    out, diags = mr.translate_config(config)
    assert out["encoding"]["otype"] == "HashGrid"  # alias normalized
    assert out["network"]["activation"] == "ReLU"
    assert out["loss"] == {"otype": "L2"}
    assert out["optimizer"] == {"otype": "Adam"}
    assert not [d for d in diags if d.severity == mr.Severity.ERROR]
