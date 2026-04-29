"""Tests for tools/migrate_tcnn.py — the migration CLI MVP."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tools"))

import migrate_rules as mr  # noqa: E402
import migrate_tcnn  # noqa: E402


# ---------------------------------------------------------------------------
# translate_source — pure-function entry, no I/O
# ---------------------------------------------------------------------------


def _severities(diags):
    return [d.severity for d in diags]


def test_no_tinycudann_import_is_a_no_op():
    src = dedent("""\
        import torch
        x = torch.randn(3)
    """)
    out, diags = migrate_tcnn.translate_source(src)
    assert out == src
    # One INFO note explaining there was nothing to do.
    assert mr.Severity.INFO in _severities(diags)


def test_import_tinycudann_as_tcnn_gets_rewritten():
    src = "import tinycudann as tcnn\n"
    out, diags = migrate_tcnn.translate_source(src)
    assert "import tiny_metal_nn as tmnn" in out
    assert "tinycudann" not in out
    # libcst-based rewriter does not surface line numbers (the libcst
    # nodes carry whitespace ownership rather than absolute positions);
    # the diagnostic message itself describes which import was changed.
    info = [
        d for d in diags
        if d.severity == mr.Severity.INFO
        and "import tiny_metal_nn as tmnn" in d.message
    ]
    assert info, "expected an INFO diagnostic for the import rewrite"


def test_bare_import_tinycudann_also_rewritten():
    src = "import tinycudann\n"
    out, diags = migrate_tcnn.translate_source(src)
    assert "import tiny_metal_nn as tmnn" in out
    # Even without an explicit alias, `tinycudann` was the implicit binding,
    # so the rewriter should have flagged it.
    assert any(
        d.severity == mr.Severity.INFO and "tinycudann" in d.message
        for d in diags
    )


def test_from_tinycudann_import_emits_warning():
    src = "from tinycudann import NetworkWithInputEncoding\n"
    _, diags = migrate_tcnn.translate_source(src)
    warns = [d for d in diags if d.severity == mr.Severity.WARNING]
    assert warns, "expected a WARNING for `from tinycudann import ...`"
    assert "from-import" in warns[0].message


def test_tcnn_network_with_input_encoding_call_emits_warning_with_suggestion():
    src = dedent("""\
        import tinycudann as tcnn
        model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3, n_output_dims=1,
            encoding_config={}, network_config={},
        )
    """)
    _, diags = migrate_tcnn.translate_source(src)
    warns = [
        d for d in diags
        if d.severity == mr.Severity.WARNING
        and "NetworkWithInputEncoding" in d.message
    ]
    assert warns
    assert warns[0].suggestion
    assert "tmnn.Trainer.from_config" in warns[0].suggestion
    assert "training_step" in warns[0].suggestion


def test_training_loop_pattern_is_rewritten():
    """Stage 11.8: the canonical 5-line training-loop body is replaced
    with a single ``loss = trainer.training_step(x, y)`` call, not just
    flagged with a WARNING. Comments / blank lines outside the loop body
    survive the libcst roundtrip."""
    src = dedent("""\
        import tinycudann as tcnn
        model = tcnn.NetworkWithInputEncoding(3, 1, {}, {})

        for x, y in loader:
            output = model(x)
            loss = ((output - y) ** 2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    """)
    out, diags = migrate_tcnn.translate_source(src)

    # Source rewritten — the training-loop body collapses to one line.
    assert "trainer.training_step(x, y)" in out
    assert "loss.backward()" not in out
    assert "optimizer.step()" not in out
    assert "optimizer.zero_grad()" not in out

    # An INFO diagnostic explains the rewrite; no WARNING about the loop
    # because it was handled mechanically.
    info = [
        d for d in diags
        if d.severity == mr.Severity.INFO and "training loop" in d.message
    ]
    assert info, "expected an INFO diagnostic for the loop rewrite"
    loop_warns = [
        d for d in diags
        if d.severity == mr.Severity.WARNING and "training loop" in d.message
    ]
    assert not loop_warns, "loop was mechanically rewritten; no WARNING expected"


def test_libcst_preserves_comments_and_blank_lines():
    """Stage 11.8 contract: ast.unparse loses comments / blank lines;
    libcst preserves them. This test pins that property — if a future
    change reverts to ast roundtrip, this fails."""
    src = dedent("""\
        # File header — stays put.
        import torch
        import tinycudann as tcnn

        # Note about config.
        ENC = {"otype": "HashGrid"}
    """)
    out, _ = migrate_tcnn.translate_source(src)
    # Comments survive.
    assert "# File header — stays put." in out
    assert "# Note about config." in out
    # Blank line between imports and config block survives.
    lines = out.splitlines()
    blank_idx = [i for i, line in enumerate(lines) if line.strip() == ""]
    assert blank_idx, "expected at least one blank line preserved"


def test_for_loop_without_backward_does_not_falsely_match():
    """A plain inference loop (no .backward()) is not a training loop."""
    src = dedent("""\
        import tinycudann as tcnn
        for x in batches:
            y = model(x)
            output_collector.append(y)
    """)
    _, diags = migrate_tcnn.translate_source(src)
    loop_warns = [
        d for d in diags
        if d.severity == mr.Severity.WARNING and "training loop" in d.message
    ]
    assert not loop_warns


def test_translate_handles_tcnn_aliased_to_arbitrary_name():
    """User who wrote `import tinycudann as ngp` still has tcnn calls
    detected (rewriter follows the alias binding)."""
    src = dedent("""\
        import tinycudann as ngp
        model = ngp.NetworkWithInputEncoding(3, 1, {}, {})
    """)
    out, diags = migrate_tcnn.translate_source(src)
    # Import got renamed; calls flagged.
    assert "import tiny_metal_nn as tmnn" in out
    assert any(
        d.severity == mr.Severity.WARNING and "NetworkWithInputEncoding" in d.message
        for d in diags
    )


def test_syntax_error_in_input_propagates():
    with pytest.raises(SyntaxError):
        migrate_tcnn.translate_source("def(): pass")


# ---------------------------------------------------------------------------
# main(argv) — CLI integration
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "input.py"
    p.write_text(dedent(text), encoding="utf-8")
    return p


def test_cli_check_returns_zero_when_no_warnings(tmp_path):
    src = _write(tmp_path, "import torch\nx = 1\n")
    rc = migrate_tcnn.main([str(src), "--check"])
    assert rc == 0


def test_cli_check_returns_one_when_warnings_present(tmp_path):
    src = _write(tmp_path, """\
        import tinycudann as tcnn
        model = tcnn.NetworkWithInputEncoding(3, 1, {}, {})
    """)
    rc = migrate_tcnn.main([str(src), "--check"])
    assert rc == 1


def test_cli_output_writes_translated_file(tmp_path):
    src = _write(tmp_path, "import tinycudann as tcnn\n")
    out_path = tmp_path / "translated.py"
    rc = migrate_tcnn.main([str(src), "--output", str(out_path)])
    # Import-only file → only INFO diagnostics → exit 0
    assert rc == 0
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "import tiny_metal_nn as tmnn" in text


def test_cli_missing_input_returns_three(tmp_path):
    rc = migrate_tcnn.main([str(tmp_path / "does_not_exist.py"), "--check"])
    assert rc == 3


def test_cli_syntax_error_returns_two(tmp_path):
    src = _write(tmp_path, "def(): pass\n")
    rc = migrate_tcnn.main([str(src), "--check"])
    assert rc == 2
