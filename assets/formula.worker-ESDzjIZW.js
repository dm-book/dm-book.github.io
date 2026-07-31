(function(){"use strict";const p="314.0.7";var c=`import json
import io
import random
import tokenize
import re
import sys

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    rationalize,
    standard_transformations,
)

MAX_FORMULA_LENGTH = 800
PARSE_ERROR = "I could not parse this as a formula."
FORMULA_CHAR_PATTERN = re.compile(r"^[A-Za-z0-9_+\\-*/^().,\\s!]+$")
TRANSFORMATIONS = standard_transformations + (implicit_multiplication, implicit_application, convert_xor, rationalize)
GLOBAL_DICT = {
    "__builtins__": {},
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Rational": sp.Rational,
    "Float": sp.Float,
    "factorial": sp.factorial,
}
FUNCTIONS = {
    "binomial": sp.binomial,
    "choose": sp.binomial,
    "factorial": sp.factorial,
    "sqrt": sp.sqrt,
    "floor": sp.floor,
    "ceil": sp.ceiling,
    "ceiling": sp.ceiling,
    "abs": sp.Abs,
    "max": sp.Max,
    "min": sp.Min,
    "log": sp.log,
    "ln": sp.log,
    "log2": lambda value: sp.log(value, 2),
    "log10": lambda value: sp.log(value, 10),
    "exp": sp.exp,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
}
# Greek letters stay single symbols; \`lambda\` is a Python keyword, and SymPy spells the symbol \`lamda\`.
GREEK_LETTERS = (
    "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda lamda mu nu xi omicron "
    "rho sigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Lamda Xi Pi Sigma Upsilon Phi Psi Omega"
).split()
RENAMED_WORDS = {"lambda": "lamda", "Lambda": "Lamda"}
KNOWN_WORDS = [*FUNCTIONS, *GREEK_LETTERS, "pi"]
# Other names are products of letters (qt = q*t, x1 = one symbol); known words are matched longest first.
WORD_PATTERN = re.compile("|".join(sorted(KNOWN_WORDS, key=len, reverse=True)) + "|[A-Za-z]")
NAME_OR_NUMBER_PATTERN = re.compile(r"(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?|[A-Za-z][A-Za-z0-9_]*")
UNICODE_REPLACEMENTS = {
    "−": "-", "–": "-", "—": "-", "×": "*", "⋅": "*", "·": "*", "÷": "/", "²": "^2", "³": "^3",
    "π": " pi ", "√": " sqrt ", "⌊": " floor(", "⌋": ")", "⌈": " ceiling(", "⌉": ")",
}
LATEX_COMMANDS = {
    "cdot": "*", "times": "*", "div": "/", "lfloor": " floor(", "rfloor": ")", "lceil": " ceiling(", "rceil": ")",
    "lvert": " abs(", "rvert": ")", "left": "", "right": "", "bigl": "", "bigr": "", "Bigl": "", "Bigr": "",
    "big": "", "Big": "", "displaystyle": "", "quad": " ", "qquad": " ",
}
LATEX_ARGUMENT_COMMAND = re.compile(r"\\\\([dtc]?frac|[dt]?binom|sqrt)(?![A-Za-z])")
# Exact values are compared at nonnegative integer points, the book's domain, when algebra alone is inconclusive.
SAMPLE_POINTS = 10
SAMPLE_ATTEMPTS = 60
SAMPLE_DIGITS = 30
SAMPLE_TOLERANCE = sp.Float("1e-20")
# Bounds exponents, factorial arguments and digits of powers, so that exact values stay cheap to compute.
SAMPLE_SIZE_LIMIT = 10000


class FormulaParseError(ValueError):
    pass


def strip_math_delimiters(value: str) -> str:
    text = value.strip()
    wrappers = [(r"\\(", r"\\)"), (r"\\[", r"\\]"), ("$$", "$$"), ("$", "$")]
    changed = True
    while changed:
        changed = False
        for left, right in wrappers:
            if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
                text = text[len(left) : len(text) - len(right)].strip()
                changed = True
    return text


def read_latex_argument(text: str, index: int):
    while index < len(text) and text[index].isspace():
        index += 1
    if index == len(text) or text[index] == "}":
        raise FormulaParseError(PARSE_ERROR)
    if text[index] != "{":
        command = re.match(r"\\\\[A-Za-z]+", text[index:])
        end = index + (len(command.group()) if command else 1)
        return text[index:end], end
    depth = 0
    for end in range(index, len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            return text[index + 1 : end], end + 1
    raise FormulaParseError(PARSE_ERROR)


def expand_latex_arguments(text: str) -> str:
    # Brace matching keeps nested arguments such as \\frac{2^{n}}{3} intact; inner commands expand on later passes.
    while match := LATEX_ARGUMENT_COMMAND.search(text):
        command, index = match.group(1), match.end()
        root = re.compile(r"\\s*\\[([^\\]]*)\\]").match(text, index) if command == "sqrt" else None
        if root:
            index = root.end()
        first, index = read_latex_argument(text, index)
        if command == "sqrt":
            replacement = f"(({first})^(1/({root.group(1)})))" if root else f"sqrt({first})"
        else:
            second, index = read_latex_argument(text, index)
            replacement = f"binomial({first},{second})" if command.endswith("binom") else f"(({first})/({second}))"
        text = text[: match.start()] + replacement + text[index:]
    return text


def replace_latex_command(match) -> str:
    name = match.group(1)
    if name in LATEX_COMMANDS:
        return LATEX_COMMANDS[name]
    return f" {name}" if name in KNOWN_WORDS else match.group()


def split_name(match) -> str:
    name = match.group()
    if not name[0].isalpha():
        return name
    if "_" in name or name in FUNCTIONS:
        return f" {name} "
    pieces = []
    for letters, digits in re.findall(r"([A-Za-z]+)(\\d*)", name):
        words = [RENAMED_WORDS.get(word, word) for word in WORD_PATTERN.findall(letters)]
        if digits and (words[-1] in FUNCTIONS or words[-1] == "pi"):
            words.append(digits)
        else:
            words[-1] += digits
        pieces += words
    text = pieces[0]
    for previous, piece in zip(pieces, pieces[1:]):
        text += (" " if previous in FUNCTIONS else "*") + piece
    return f" {text} "


def normalize_formula_text(value) -> str:
    if not isinstance(value, (str, int, float)):
        raise FormulaParseError("Enter a formula.")
    text = strip_math_delimiters(str(value))
    if not text:
        raise FormulaParseError("Enter a formula.")
    if len(text) > MAX_FORMULA_LENGTH:
        raise FormulaParseError("The formula is too long.")
    for character, replacement in UNICODE_REPLACEMENTS.items():
        text = text.replace(character, replacement)
    text = re.sub(r"\\\\[,;:! ]", " ", text)
    text = expand_latex_arguments(text)
    # Numeric subscripts join the name (x_{1} is x1, \\log_2 is log2); other subscripts stay in it (a_n).
    text = re.sub(r"(?<=[A-Za-z])_\\{\\s*(\\w+)\\s*\\}", r"_\\1", text)
    text = re.sub(r"(?<=[A-Za-z])_(\\d+)", r"\\1", text)
    text = re.sub(r"\\\\([A-Za-z]+)", replace_latex_command, text)
    text = re.sub(r"\\|([^|]*)\\|", r" abs(\\1)", text)
    text = text.replace("{", "(").replace("}", ")")
    if "\\\\" in text or not FORMULA_CHAR_PATTERN.fullmatch(text):
        raise FormulaParseError(PARSE_ERROR)
    text = NAME_OR_NUMBER_PATTERN.sub(split_name, text)
    # parse_expr evaluates generated Python. Accept mathematical tokens only;
    # attribute access, strings and Python's internal names are never formulas.
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.string == "." or (token.type == tokenize.NAME and
                                      (token.string.startswith("_") or "__" in token.string)):
                raise FormulaParseError(PARSE_ERROR)
    except (tokenize.TokenError, IndentationError) as error:
        raise FormulaParseError(PARSE_ERROR) from error
    return text


def parse_formula(value):
    text = normalize_formula_text(value)
    local_dict = {"pi": sp.pi, **FUNCTIONS}
    for name in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text):
        if name not in local_dict and name not in GLOBAL_DICT:
            local_dict[name] = sp.Symbol(name)
    try:
        parsed = parse_expr(
            text,
            local_dict=local_dict,
            global_dict=GLOBAL_DICT,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as error:
        raise FormulaParseError(PARSE_ERROR) from error
    if not isinstance(parsed, sp.Expr):
        raise FormulaParseError(PARSE_ERROR)
    return parsed


def preview_formula(answer):
    try:
        return {"latex": sp.latex(parse_formula(answer))}
    except FormulaParseError as error:
        return {"formatError": True, "message": str(error)}


def sample(expression, point):
    """The exact value at an integer point, or None where it is undefined or too large to compute."""
    try:
        # Exact n^(n^n) or (n!)! could exhaust memory, so sizes are estimated numerically first, innermost first.
        for node in sp.postorder_traversal(expression):
            if isinstance(node, (sp.factorial, sp.binomial)):
                sizes = [abs(argument.evalf(15, subs=point)) for argument in node.args]
            elif isinstance(node, (sp.Pow, sp.exp)) and node.as_base_exp()[0] not in (1, -1):
                base, exponent = (abs(part.evalf(15, subs=point)) for part in node.as_base_exp())
                sizes = [exponent, exponent * abs(sp.log(base, 10)) if base else 0]
            else:
                continue
            if not all(size < SAMPLE_SIZE_LIMIT for size in sizes):
                return None
        value = expression.xreplace(point)
    except Exception:
        return None
    return value if value.is_number and value.is_finite else None


def same_value(left, right) -> bool:
    if left.is_Rational and right.is_Rational:
        return left == right
    scale = max(abs(left.evalf(SAMPLE_DIGITS)), abs(right.evalf(SAMPLE_DIGITS)), 1)
    return bool(abs((left - right).evalf(SAMPLE_DIGITS)) <= SAMPLE_TOLERANCE * scale)


def agree_on_integers(answer, expected) -> bool:
    # Every variable varies independently, so a variable the expected formula lacks must not change the value.
    symbols = sorted(answer.free_symbols | expected.free_symbols, key=str)
    generator = random.Random(0)
    tried = set()
    agreed = 0
    for attempt in range(SAMPLE_ATTEMPTS):
        # Ranges cycle through 0..1, 0..8, ..., 0..4096: small edge cases and values far beyond them.
        values = tuple(generator.randint(0, 8 ** (attempt % 5)) for _ in symbols)
        if values in tried:
            continue
        tried.add(values)
        point = {symbol: sp.Integer(value) for symbol, value in zip(symbols, values)}
        left, right = sample(answer, point), sample(expected, point)
        if left is None or right is None:
            continue
        if not same_value(left, right):
            return False
        agreed += 1
        if agreed == SAMPLE_POINTS:
            return True
    return False


def equivalent(answer, expected) -> bool:
    difference = sp.simplify(sp.together(answer - expected))
    if difference == 0 or difference.equals(0) is True:
        return True
    symbols = difference.free_symbols
    # Sampling cannot help a constant, and a nonzero rational function vanishes at finitely many points.
    if not symbols or difference.is_rational_function(*symbols):
        return False
    return agree_on_integers(answer, expected)


def check_answer(answer, expected):
    try:
        answer_expr = parse_formula(answer)
        passed = equivalent(answer_expr, parse_formula(expected))
    except FormulaParseError as error:
        return {
            "passed": False,
            "message": str(error) or PARSE_ERROR,
            "formatError": True,
        }
    return {"passed": passed, "message": "Correct." if passed else "Not yet.", "latex": sp.latex(answer_expr)}


def main():
    payload = json.loads(sys.stdin.read() or "{}")
    print(json.dumps(check_answer(payload.get("answer"), payload.get("expected"))))


if __name__ == "__main__":
    main()
`,m=`"""Run an exported author checker with the student's response passed as data."""

import math


def check_string(source, answer):
    namespace = {"__name__": "string_checker"}
    exec(source, namespace)
    try:
        result = namespace["check"](answer)
    except (ValueError, IndexError):
        return {"passed": False, "score": 0, "formatError": True,
                "message": "Could not read this answer. Check the format requested in the problem."}
    verdict, message = (result[0], result[1] if len(result) > 1 else "") if isinstance(result, tuple) else (result, "")
    score = float(verdict)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid checker score")
    return {"passed": score > 0, "score": score, "message": str(message)}
`;const t=self,s=new URL(`/math-runtime/${p}/`,t.location.origin).href;let d,i=!1,l=!1;function u(){return d??=(async()=>{const{loadPyodide:e}=await import(`${s}pyodide.mjs`);return e({indexURL:s})})()}async function f(e,n){if(n.includes("numpy")&&await e.loadPackage("numpy"),n.includes("networkx")&&!l){const r=await fetch(`${s}pyodide-lock.json`);if(!r.ok)throw new Error("Package list unavailable");const a=await r.json(),o=await fetch(`${s}${a.packages.networkx.file_name}`);if(!o.ok)throw new Error("NetworkX unavailable");e.unpackArchive(await o.arrayBuffer(),"wheel",{extractDir:e.runPython(`import sysconfig
sysconfig.get_paths()['purelib']`)}),l=!0}}t.onmessage=async({data:e})=>{try{const n=await u(),r=e.operation==="string";r?await f(n,e.packages):i||(await n.loadPackage("sympy"),i=!0);const a=n.toPy({__name__:"site_checker",answer:e.answer,expected:e.expected,source:e.source});try{n.runPython(r?m:c,{globals:a}),t.postMessage({type:"ready"});const o=r?"check_string(source, answer)":e.operation==="preview"?"preview_formula(answer)":"check_answer(answer, expected)",_=n.runPython(`import json
json.dumps(${o})`,{globals:a});t.postMessage({type:"result",result:JSON.parse(_)})}finally{a.destroy()}}catch{t.postMessage({type:"error"})}}})();
