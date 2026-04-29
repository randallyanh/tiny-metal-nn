#!/usr/bin/env python3
"""tools/migrate_tcnn.py — tcnn → tmnn migration CLI.

Mechanical translation of Python source that imports tinycudann to tmnn,
backed by `libcst` so comments / blank lines / formatting survive the
roundtrip (`ast.unparse` would flatten them).

Mechanical rewrites:
    - ``import tinycudann [as X]`` → ``import tiny_metal_nn as tmnn``
    - Training-loop body (``output = model(x); loss = ((output - y) ** 2)
      .mean(); loss.backward(); optimizer.step(); optimizer.zero_grad()``)
      → ``loss = trainer.training_step(x, y)``

Diagnostics-only (still flagged, not auto-rewritten):
    - ``tcnn.NetworkWithInputEncoding(...)`` calls — extracting the right
      args into a ``tmnn.Trainer.from_config({...})`` call requires
      cross-referencing the surrounding optimizer + batch-size literals
      and is brittle to encode mechanically. Skill workflow handles it
      (006 v2 §4.2). v1.x roadmap may add a more aggressive rewriter.
    - ``from tinycudann import X`` — symbol-by-symbol mirrors don't exist
      on tmnn's side; the user must restructure to a single ``Trainer``
      class.

Usage:
    tmnn-migrate train.py --check
    tmnn-migrate train.py --diff
    tmnn-migrate train.py --output train_tmnn.py

Exit codes:
    0  — every change applied; no manual work pending
    1  — some segments need manual review (diagnostics list each one)
    2  — input file could not be parsed
    3  — argument or filesystem error
"""

from __future__ import annotations

import argparse
import difflib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import libcst as cst
import libcst.matchers as m

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import migrate_rules as mr  # noqa: E402


# ---------------------------------------------------------------------------
# Diagnostic accumulation
# ---------------------------------------------------------------------------


@dataclass
class CliDiagnostic:
    severity: mr.Severity
    line: Optional[int]
    message: str
    suggestion: Optional[str] = None

    def render(self, source_path: Path) -> str:
        loc = f"{source_path}" + (f":{self.line}" if self.line else "")
        head = f"{loc}: {self.severity.value}: {self.message}"
        if self.suggestion:
            return head + f"\n        suggest: {self.suggestion}"
        return head


# ---------------------------------------------------------------------------
# CST transformer — handles imports + training-loop rewrites
# ---------------------------------------------------------------------------


# Match a top-level expression statement that wraps a `<expr>.<method>()`
# call, e.g., `optimizer.step()`. Used to detect loss.backward(),
# optimizer.step(), optimizer.zero_grad() in a training-loop body.
def _is_method_call_stmt(stmt: cst.BaseStatement, method: str) -> bool:
    if not isinstance(stmt, cst.SimpleStatementLine):
        return False
    if len(stmt.body) != 1:
        return False
    expr = stmt.body[0]
    if not isinstance(expr, cst.Expr):
        return False
    return m.matches(
        expr.value,
        m.Call(func=m.Attribute(attr=m.Name(value=method))),
    )


# Match `output = model(x)` (single assign target, RHS is `<name>(<single arg>)`).
# Returns (target_name, model_name, arg_node) or None.
def _match_forward_assign(
    stmt: cst.BaseStatement,
) -> Optional[Tuple[str, str, cst.BaseExpression]]:
    if not isinstance(stmt, cst.SimpleStatementLine):
        return None
    if len(stmt.body) != 1:
        return None
    assign = stmt.body[0]
    if not isinstance(assign, cst.Assign) or len(assign.targets) != 1:
        return None
    target = assign.targets[0].target
    if not isinstance(target, cst.Name):
        return None
    rhs = assign.value
    if not isinstance(rhs, cst.Call) or not isinstance(rhs.func, cst.Name):
        return None
    if len(rhs.args) != 1:
        return None
    return target.value, rhs.func.value, rhs.args[0].value


# Match `loss = ((output - target) ** 2).mean()` (canonical MSE).
# Returns (loss_name, target_node) on match. Drill in directly via
# isinstance checks — libcst's matchers module doesn't include a
# universal "any expression" pattern, but isinstance walks are clean.
def _match_mse_loss_assign(
    stmt: cst.BaseStatement, output_name: str
) -> Optional[Tuple[str, cst.BaseExpression]]:
    if not isinstance(stmt, cst.SimpleStatementLine):
        return None
    if len(stmt.body) != 1:
        return None
    assign = stmt.body[0]
    if not isinstance(assign, cst.Assign) or len(assign.targets) != 1:
        return None
    target = assign.targets[0].target
    if not isinstance(target, cst.Name):
        return None
    # RHS must be `<something>.mean()`.
    rhs = assign.value
    if not isinstance(rhs, cst.Call):
        return None
    if not isinstance(rhs.func, cst.Attribute):
        return None
    if not (isinstance(rhs.func.attr, cst.Name) and rhs.func.attr.value == "mean"):
        return None
    inner = rhs.func.value  # the expression `.mean()` is called on
    # Expect `(... ** 2)` — possibly wrapped in parens (libcst represents
    # parens as Atom-level `lpar`/`rpar` lists, the BinaryOperation itself
    # is the inner node).
    if not isinstance(inner, cst.BinaryOperation):
        return None
    if not isinstance(inner.operator, cst.Power):
        return None
    if not (
        isinstance(inner.right, cst.Integer) and inner.right.value == "2"
    ):
        return None
    diff = inner.left
    if not isinstance(diff, cst.BinaryOperation):
        return None
    if not isinstance(diff.operator, cst.Subtract):
        return None
    if not (
        isinstance(diff.left, cst.Name) and diff.left.value == output_name
    ):
        return None
    return target.value, diff.right


class _TcnnUsageRewriter(cst.CSTTransformer):
    """libcst transformer for the tcnn → tmnn source rewrite.

    Rewrites mechanical pieces (imports + the canonical 5-line training
    loop). Flags non-mechanical pieces (NetworkWithInputEncoding calls,
    star imports) via diagnostics for the human / Claude skill to handle.
    """

    def __init__(self, source: str) -> None:
        self.source = source
        self.diagnostics: List[CliDiagnostic] = []
        self.tcnn_aliases: set[str] = set()
        self.rewrote_any_import = False
        self.rewrote_any_loop = False

    # ── helpers ──────────────────────────────────────────────────────────

    def _line_of(self, node: cst.CSTNode, default: int = 0) -> int:
        # libcst doesn't store line numbers natively; the transformer is
        # walked top-down so we can approximate via source.find on the
        # original. For diagnostics this is good enough.
        return default

    # ── imports ──────────────────────────────────────────────────────────

    def leave_Import(
        self, original_node: cst.Import, updated_node: cst.Import
    ) -> cst.BaseSmallStatement:
        new_names: List[cst.ImportAlias] = []
        rewrote = False
        for alias in updated_node.names:
            name = self._alias_name(alias.name)
            if name == "tinycudann":
                bound = (
                    alias.asname.name.value
                    if alias.asname
                    and isinstance(alias.asname.name, cst.Name)
                    else "tinycudann"
                )
                self.tcnn_aliases.add(bound)
                new_names.append(
                    cst.ImportAlias(
                        name=cst.Name("tiny_metal_nn"),
                        asname=cst.AsName(name=cst.Name("tmnn")),
                    )
                )
                rewrote = True
                self.diagnostics.append(
                    CliDiagnostic(
                        severity=mr.Severity.INFO,
                        line=None,
                        message=(
                            f"rewrote `import tinycudann"
                            f"{' as ' + bound if bound != 'tinycudann' else ''}`"
                            f" to `import tiny_metal_nn as tmnn`"
                        ),
                    )
                )
            else:
                new_names.append(alias)
        if rewrote:
            self.rewrote_any_import = True
        return updated_node.with_changes(names=new_names)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> Optional[bool]:
        # Detect `from tinycudann import ...` but do not auto-rewrite; the
        # symbol surface differs structurally between tcnn and tmnn.
        module = self._module_name(node.module)
        if module == "tinycudann":
            self.diagnostics.append(
                CliDiagnostic(
                    severity=mr.Severity.WARNING,
                    line=None,
                    message=(
                        "from-import of tinycudann symbols is not mechanically "
                        "rewritable; tmnn exposes a single Trainer class, not "
                        "per-symbol mirrors."
                    ),
                    suggestion=(
                        "Replace with `import tiny_metal_nn as tmnn` and "
                        "use `tmnn.Trainer.from_config(...)`."
                    ),
                )
            )
        return None

    @staticmethod
    def _alias_name(node: cst.BaseExpression) -> str:
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            return (
                _TcnnUsageRewriter._alias_name(node.value)
                + "."
                + node.attr.value
            )
        return ""

    @staticmethod
    def _module_name(node: Optional[cst.BaseExpression]) -> str:
        if node is None:
            return ""
        return _TcnnUsageRewriter._alias_name(node)

    # ── tcnn.* call sites — diagnostic only, no rewrite ──────────────────

    def leave_Call(
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.BaseExpression:
        func = updated_node.func
        if isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name):
            if func.value.value in self.tcnn_aliases:
                self._handle_tcnn_call(func.attr.value)
        return updated_node

    def _handle_tcnn_call(self, attr: str) -> None:
        if attr == "NetworkWithInputEncoding":
            self.diagnostics.append(
                CliDiagnostic(
                    severity=mr.Severity.WARNING,
                    line=None,
                    message=(
                        "tcnn.NetworkWithInputEncoding(...) does not have a "
                        "1-to-1 tmnn equivalent: tmnn fuses the trainer + "
                        "loss + optimizer into a single from_config(...) "
                        "call."
                    ),
                    suggestion=(
                        "Replace with: trainer = tmnn.Trainer.from_config("
                        "{\"encoding\": <enc>, \"network\": <net>, "
                        "\"loss\": {\"otype\": \"L2\"}, "
                        "\"optimizer\": {\"otype\": \"Adam\", "
                        "\"learning_rate\": <lr>}, \"batch_size\": <N>}, "
                        "n_input=<n_in>, n_output=<n_out>). "
                        "Then drive training via trainer.training_step(x, y) "
                        "and inference via trainer.inference(x)."
                    ),
                )
            )
        elif attr in {"Network", "Encoding"}:
            self.diagnostics.append(
                CliDiagnostic(
                    severity=mr.Severity.WARNING,
                    line=None,
                    message=(
                        f"tcnn.{attr}(...) used standalone — tmnn does not "
                        f"expose Network / Encoding as separate Python "
                        f"classes; both live inside tmnn.Trainer."
                    ),
                    suggestion=(
                        "Combine your encoding + network configs into a "
                        "single tmnn.Trainer.from_config(...) call."
                    ),
                )
            )

    # ── training-loop body rewrite ───────────────────────────────────────
    #
    # Pattern (consecutive statements inside the body of a `for` loop):
    #     output = model(x)                       # forward
    #     loss = ((output - y) ** 2).mean()       # canonical MSE
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()                   # may also be optimizer.zero_grad()
    #
    # Replacement:
    #     loss = trainer.training_step(x, y)

    def leave_For(
        self, original_node: cst.For, updated_node: cst.For
    ) -> cst.BaseStatement:
        body = updated_node.body
        if not isinstance(body, cst.IndentedBlock):
            return updated_node
        new_stmts = self._rewrite_loop_body(list(body.body))
        if new_stmts is None:
            return updated_node
        self.rewrote_any_loop = True
        self.diagnostics.append(
            CliDiagnostic(
                severity=mr.Severity.INFO,
                line=None,
                message=(
                    "rewrote canonical 5-line training loop body to a single "
                    "`loss = trainer.training_step(x, y)` call (006 v2 § 10.3)"
                ),
            )
        )
        return updated_node.with_changes(
            body=body.with_changes(body=new_stmts)
        )

    def _rewrite_loop_body(
        self, stmts: List[cst.BaseStatement]
    ) -> Optional[List[cst.BaseStatement]]:
        # Find the canonical 5-statement window. We allow other statements
        # before/after; only rewrite the matched window.
        for i in range(len(stmts) - 4):
            forward = _match_forward_assign(stmts[i])
            if not forward:
                continue
            output_name, _model_name, input_arg = forward
            mse = _match_mse_loss_assign(stmts[i + 1], output_name)
            if not mse:
                continue
            loss_name, target_node = mse
            if not (
                _is_method_call_stmt(stmts[i + 2], "backward")
                and _is_method_call_stmt(stmts[i + 3], "step")
                and _is_method_call_stmt(stmts[i + 4], "zero_grad")
            ):
                continue
            # All five statements matched; replace with a single
            # `<loss_name> = trainer.training_step(<input>, <target>)`.
            replacement = cst.SimpleStatementLine(
                body=[
                    cst.Assign(
                        targets=[
                            cst.AssignTarget(target=cst.Name(loss_name))
                        ],
                        value=cst.Call(
                            func=cst.Attribute(
                                value=cst.Name("trainer"),
                                attr=cst.Name("training_step"),
                            ),
                            args=[
                                cst.Arg(value=input_arg),
                                cst.Arg(value=target_node),
                            ],
                        ),
                    )
                ],
                leading_lines=stmts[i].leading_lines
                if hasattr(stmts[i], "leading_lines")
                else (),
            )
            return stmts[:i] + [replacement] + stmts[i + 5 :]
        return None


# ---------------------------------------------------------------------------
# Top-level translation pipeline
# ---------------------------------------------------------------------------


def translate_source(source: str) -> Tuple[str, List[CliDiagnostic]]:
    """Translate `source` (Python text) and return ``(translated, diagnostics)``.

    Raises:
        SyntaxError: if the source cannot be parsed.
    """
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError as exc:
        # Re-raise as the standard Python exception so callers can rely on
        # the same `except SyntaxError:` catch as the previous AST impl.
        raise SyntaxError(str(exc)) from exc

    rewriter = _TcnnUsageRewriter(source)
    new_module = module.visit(rewriter)

    diagnostics: List[CliDiagnostic] = list(rewriter.diagnostics)

    if rewriter.rewrote_any_import or rewriter.rewrote_any_loop:
        translated = new_module.code
    else:
        translated = source
        diagnostics.append(
            CliDiagnostic(
                severity=mr.Severity.INFO,
                line=None,
                message=(
                    "no `import tinycudann` and no canonical training loop "
                    "found; nothing to translate. (Input may already be "
                    "tmnn-shaped or be a different file.)"
                ),
            )
        )

    return translated, diagnostics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_diff(original: str, translated: str, source_path: Path) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            translated.splitlines(keepends=True),
            fromfile=f"{source_path} (tcnn)",
            tofile=f"{source_path} (tmnn)",
        )
    )


def _exit_code(diagnostics: Iterable[CliDiagnostic]) -> int:
    """0 if every change is mechanical; 1 if any WARNING/ERROR remains."""
    for d in diagnostics:
        if d.severity in (mr.Severity.WARNING, mr.Severity.ERROR):
            return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tmnn-migrate",
        description=(
            "Translate Python source from tinycudann (tcnn) to "
            "tiny_metal_nn (tmnn). See docs/TCNN-MIGRATION-GUIDE.md § 10 "
            "and docs/know-how/006-python-binding-design.md v2 § 4."
        ),
    )
    parser.add_argument(
        "input", type=Path, help="Python file to translate (.py)."
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Run analysis only; do not write anything. Diagnostics go "
             "to stderr; exit code reports manual-work pending.",
    )
    parser.add_argument(
        "--diff", action="store_true",
        help="Print a unified diff to stdout. Combine with --output to "
             "also write the translated source.",
    )
    parser.add_argument(
        "--output", type=Path, metavar="OUT.py",
        help="Write the translated source to OUT.py. If omitted and "
             "neither --check nor --diff is set, prints translated "
             "source to stdout.",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"error: not a file: {args.input}", file=sys.stderr)
        return 3

    source = args.input.read_text(encoding="utf-8")

    try:
        translated, diagnostics = translate_source(source)
    except SyntaxError as exc:
        print(f"{args.input}: syntax error: {exc.msg}", file=sys.stderr)
        return 2

    for d in diagnostics:
        print(d.render(args.input), file=sys.stderr)

    if args.diff:
        sys.stdout.write(_format_diff(source, translated, args.input))

    if args.output:
        args.output.write_text(translated, encoding="utf-8")
        print(f"wrote: {args.output}", file=sys.stderr)
    elif not args.check and not args.diff:
        sys.stdout.write(translated)

    return _exit_code(diagnostics)


if __name__ == "__main__":
    raise SystemExit(main())
