"""Conservative MathWriting-to-HME100K LaTeX normalization.

The checkpoint's output layer has exactly the HME100K vocabulary.  This module
therefore rejects an expression whenever a token cannot be represented instead
of silently mapping it to an unknown token.
"""

import re
from typing import Iterable, List, Optional, Sequence, Set, Tuple


COMMAND_ALIASES = {
    r"\dfrac": r"\frac",
    r"\tfrac": r"\frac",
    r"\le": r"\leq",
    r"\ge": r"\geq",
    r"\ne": r"\neq",
    r"\to": r"\rightarrow",
    r"\emptyset": r"\varnothing",
    r"\vert": r"\mid",
}

DROP_COMMANDS = {r"\left", r"\right", r"\displaystyle", r"\textstyle"}
FUNCTIONS = ("arccos", "arcsin", "arctan", "sin", "cos", "tan", "cot", "ln", "log", "lim")
TOKEN_RE = re.compile(r"\\[A-Za-z]+|\\.|[A-Za-z0-9]|[^\s]")


def load_vocabulary(path: str) -> Set[str]:
    with open(path, "r", encoding="utf-8") as stream:
        return {line.strip() for line in stream if line.strip()}


def _pre_normalize(label: str) -> str:
    value = label.strip().replace("\u2212", "-").replace("\u00d7", r"\times")
    value = re.sub(r"\\operatorname\s*\{\s*(sin|cos|tan|cot|ln|log|lim)\s*\}", r"\\\1", value)
    value = re.sub(r"\\mathrm\s*\{\s*d\s*\}", "d", value)
    value = re.sub(r"\\text\s*\{\s*d\s*\}", "d", value)
    # Some MathWriting labels contain plain function names, e.g. sin(x).
    for name in FUNCTIONS:
        value = re.sub(
            r"(?<![A-Za-z\\])" + name + r"(?=\s*[_({A-Za-z0-9\\])",
            r"\\" + name,
            value,
        )
    return value


def tokenize_latex(label: str) -> List[str]:
    raw_tokens = []
    for token in TOKEN_RE.findall(_pre_normalize(label)):
        token = COMMAND_ALIASES.get(token, token)
        if token not in DROP_COMMANDS:
            raw_tokens.append(token)
    # HME100K canonical captions always brace scripts. MathWriting also permits
    # x^2 and x_i; make those equivalent to x^{2} and x_{i}.
    tokens = []
    index = 0
    while index < len(raw_tokens):
        token = raw_tokens[index]
        tokens.append(token)
        if token in ("^", "_") and index + 1 < len(raw_tokens) and raw_tokens[index + 1] != "{":
            tokens.extend(("{", raw_tokens[index + 1], "}"))
            index += 2
        else:
            index += 1
    return tokens


def is_balanced(tokens: Sequence[str]) -> bool:
    pairs = {"}": "{", "]": "[", ")": "("}
    stack = []
    for token in tokens:
        if token in ("{", "[", "("):
            stack.append(token)
        elif token in pairs:
            if not stack or stack.pop() != pairs[token]:
                return False
    return not stack


def has_valid_script_syntax(tokens: Sequence[str]) -> bool:
    for index, token in enumerate(tokens):
        if token in ("^", "_"):
            if index + 2 >= len(tokens) or tokens[index + 1] != "{":
                return False
    return True


def normalize_and_tokenize(
    label: str, vocabulary: Set[str], max_tokens: int = 200
) -> Tuple[Optional[List[str]], Optional[str]]:
    tokens = tokenize_latex(label)
    if not tokens:
        return None, "empty"
    if len(tokens) > max_tokens:
        return None, "too_long"
    if not is_balanced(tokens):
        return None, "unbalanced_brackets"
    if not has_valid_script_syntax(tokens):
        return None, "invalid_script_syntax"
    oov = sorted({token for token in tokens if token not in vocabulary})
    if oov:
        return None, "oov:" + ",".join(oov)
    return tokens, None


def canonical_label(tokens: Iterable[str]) -> str:
    return " ".join(tokens)


def _has_derivative(compact: str) -> bool:
    return any(marker in compact for marker in (r"\partial", r"\prime", r"\dot", r"\ddot")) or bool(
        re.search(r"\\frac\{d[^}]*\}\{d[^}]*\}", compact)
    )


def categorize_formula(label: str) -> Optional[str]:
    compact = re.sub(r"\s+", "", label)
    flags = {
        "integral": r"\int" in compact,
        "derivative": _has_derivative(compact),
        "limit": r"\lim" in compact or bool(re.search(r"(?<![A-Za-z])lim(?=[_({A-Za-z0-9])", compact)),
        "sum_series": r"\sum" in compact,
        "trig_log_exp": any(
            marker in compact
            for marker in (
                r"\sin", r"\cos", r"\tan", r"\cot", r"\ln", r"\log",
                "sin(", "cos(", "tan(", "sin", "cos", "tan", "loga", "lna",
            )
        ),
    }
    active = [name for name, enabled in flags.items() if enabled]
    # Limits are the rarest target class in the compatible HME vocabulary.
    # Keep them in their own bucket even when sin/log also occurs.
    if flags["limit"]:
        return "limit"
    if len(active) >= 2:
        return "mixed"
    return active[0] if active else None


def latex_is_syntactically_valid(tokens: Sequence[str]) -> bool:
    """A deterministic syntax check used by the Valid-LaTeX metric."""
    return bool(tokens) and is_balanced(tokens) and has_valid_script_syntax(tokens)
