# ==========================================
# 模块 1：依赖导入与页面配置
# ==========================================
import re
import os
import time
import base64
import hashlib
import unicodedata
import json
from io import BytesIO
from typing import Any, Dict, List, Tuple, Optional
import datetime
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
st.set_page_config(page_title="学术文献智能工作台", page_icon="📚", layout="wide")


# ==========================================
# 模块 2：全局变量与 API 配置
# ==========================================
ANALYSIS_CACHE_VERSION = "20260412_history_single_image_v5"

# 论文搜索与论文精读报告均由后端统一执行；前端仅负责提交任务、状态展示与结果渲染。
# 页面状态统一按固定 3 分钟间隔自动刷新；论文搜索 30 分钟默认确认由后端独立计时完成。

# ==========================================
# 模块 4：通用展示函数
# ==========================================
def normalize_report_markdown(md_text: str) -> str:
    """
    清洗报告 markdown：
    - 去掉孤立的 0
    - 压缩多余空行
    - 统一换行
    """
    text = md_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'^\s*0\s*$', '', text, flags=re.M)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def get_adaptive_web_image_width(image_bytes: bytes) -> Optional[int]:
    """网页报告图片按原始像素自适应显示：小图不放大，大图按类型设上限。"""
    try:
        from PIL import Image
        with Image.open(BytesIO(image_bytes)) as img:
            iw, ih = img.size
    except Exception:
        return None

    if iw <= 0 or ih <= 0:
        return None

    ratio = iw / max(ih, 1)
    if iw <= 520:
        return int(iw)
    if ratio > 1.8:
        return int(min(iw, 960))
    if ratio < 0.72:
        return int(min(iw, 560))
    return int(min(iw, 760))


def render_report_with_images(report_md: str, images_dict: Dict[str, str]):
    """
    Streamlit 页面预览：
    - 识别 Markdown 图片语法
    - 其他内容正常 markdown 渲染
    """
    pattern = r'(!\[.*?\]\(.*?\))'
    parts = re.split(pattern, report_md)

    for part in parts:
        if not part or not part.strip():
            continue

        img_match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', part.strip(), flags=re.S)
        if not img_match:
            st.markdown(part)
            continue

        caption = img_match.group(1).strip()
        img_key = img_match.group(2).strip()

        matched = False
        for name, b64 in images_dict.items():
            if img_key == name or img_key in name or name in img_key:
                image_bytes = base64.b64decode(b64)
                display_width = get_adaptive_web_image_width(image_bytes)
                if display_width:
                    st.image(image_bytes, caption=caption, width=display_width)
                else:
                    st.image(image_bytes, caption=caption, use_container_width=True)
                matched = True
                break

        if not matched:
            st.markdown(part)


# ==========================================
# 模块 6：报告展示后处理
# ==========================================
TABLE_TITLE_LINE_RE = re.compile(
    r'^(?:表|table)\s*(?:\d+|[IVXLCDM]+|[一二三四五六七八九十百千万]+)(?:\s*[：:.-]|(?:\s+[-–—]\s+)|\s+).+',
    flags=re.I,
)
INLINE_MARKUP_TOKEN_RE = re.compile(
    r'(\*\*[^\n]+?\*\*|__[^\n]+?__|\$[^$\n]+\$|\\\([^\n]+?\\\)|\\\[[^\n]+?\\\]|`[^`\n]+`)',
    flags=re.S,
)
ASCII_TEXT_RE = re.compile(r'([A-Za-z0-9\.\,\:\;\-\+\=\(\)\/_%#@&\[\]\{\}<>\'\"\s]+)')
MATH_UNICODE_RANGE = "𝐀-𝟿"
FORMULA_BRACE_EXPR = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
FORMULA_COMBINING_MARKS = r'[\u0300-\u036F]*'
FORMULA_SUBSUP_PATTERN = (
    rf'(?:[_^](?:{FORMULA_BRACE_EXPR}|\\[A-Za-z]+|[A-Za-z0-9Α-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]+{FORMULA_COMBINING_MARKS}))+'
)
FORMULA_TERM_PATTERN = (
    rf'(?:'
    rf'\\[A-Za-z]+(?:{FORMULA_BRACE_EXPR})*(?:{FORMULA_SUBSUP_PATTERN})*'
    rf'|'
    rf'[Α-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]+{FORMULA_COMBINING_MARKS}(?:{FORMULA_SUBSUP_PATTERN})*'
    rf'|'
    rf'[A-Za-z][A-Za-z0-9]*{FORMULA_COMBINING_MARKS}(?:{FORMULA_SUBSUP_PATTERN})'
    rf'|'
    rf'\d+(?:\.\d+)?(?:{FORMULA_SUBSUP_PATTERN})'
    rf')'
)
AUTO_FORMULA_RUN_RE = re.compile(
    rf'{FORMULA_TERM_PATTERN}'
    rf'(?:\s*(?:=|<|>|/|\+|\*|·|×|÷|\\cdot)\s*(?:{FORMULA_TERM_PATTERN}|\d+(?:\.\d+)?))*'
    rf'(?:\s+{FORMULA_TERM_PATTERN})*'
)
GREEK_LATEX_MAP = {
    'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma', 'δ': r'\delta', 'ε': r'\epsilon',
    'θ': r'\theta', 'λ': r'\lambda', 'μ': r'\mu', 'µ': r'\mu', 'σ': r'\sigma', 'ω': r'\omega',
    'π': r'\pi', 'η': r'\eta', 'τ': r'\tau', 'φ': r'\phi', 'ψ': r'\psi', 'ρ': r'\rho',
    'ν': r'\nu', 'κ': r'\kappa', 'ξ': r'\xi', 'Δ': r'\Delta', 'Γ': r'\Gamma',
    'Λ': r'\Lambda', 'Σ': r'\Sigma', 'Π': r'\Pi', 'Ω': r'\Omega', 'Φ': r'\Phi', 'Ψ': r'\Psi',
}
SPECIAL_FORMULA_CHAR_MAP = {
    'ℒ': r'\mathcal{L}', '·': r'\cdot', '×': r'\times', '÷': r'\div',
    '⊙': r'\odot', '⊕': r'\oplus', '⊗': r'\otimes', '⊘': r'\oslash',
    '⊖': r'\ominus', '⊚': r'\circledcirc', '⊛': r'\circledast',
    '≤': r'\le', '≥': r'\ge', '≠': r'\neq', '≈': r'\approx',
    '≃': r'\simeq', '≅': r'\cong', '≡': r'\equiv', '∼': r'\sim',
    '±': r'\pm', '∞': r'\infty', '∑': r'\sum', '∏': r'\prod',
    '√': r'\sqrt', '∂': r'\partial', '∇': r'\nabla',
    '∈': r'\in', '∉': r'\notin', '∋': r'\ni', '∝': r'\propto',
    '∩': r'\cap', '∪': r'\cup', '⊂': r'\subset', '⊃': r'\supset',
    '⊆': r'\subseteq', '⊇': r'\supseteq', '∅': r'\emptyset',
    '∀': r'\forall', '∃': r'\exists',
}

LATEX_COMMAND_SPLIT_CANDIDATES = [
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta', 'lambda', 'mu', 'sigma', 'omega',
    'pi', 'eta', 'tau', 'phi', 'psi', 'rho', 'nu', 'kappa', 'xi',
    'Delta', 'Gamma', 'Lambda', 'Sigma', 'Pi', 'Omega', 'Phi', 'Psi',
    'partial', 'nabla', 'cdot', 'times', 'div', 'sum', 'prod', 'sqrt', 'hat', 'tilde', 'bar',
]
LATEX_COMMAND_SPLIT_RE = re.compile(
    r'(\\(?:' + '|'.join(sorted(LATEX_COMMAND_SPLIT_CANDIDATES, key=len, reverse=True)) + r'))(?=[A-Za-zΑ-Ωα-ωℒμµ])'
)
MATH_OPERATOR_CHARS = ''.join([
    '⊙', '⊕', '⊗', '⊘', '⊖', '⊚', '⊛',
    '≤', '≥', '≠', '≈', '≃', '≅', '≡', '∼',
    '±', '∞', '∑', '∏', '√', '∂', '∇',
    '∈', '∉', '∋', '∝', '∩', '∪', '⊂', '⊃',
    '⊆', '⊇', '∅', '∀', '∃',
])
MATH_OPERATOR_CHARS_ESC = re.escape(MATH_OPERATOR_CHARS)
MATH_VARIABLE_CHARS = 'abcdefghijklmnopqrstuvwxyz'
SUPERSCRIPT_CHAR_MAP = {
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    '⁺': '+', '⁻': '-', '⁼': '=', '⁽': '(', '⁾': ')',
    'ⁿ': 'n', 'ⁱ': 'i',
}
SUBSCRIPT_CHAR_MAP = {
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
    '₊': '+', '₋': '-', '₌': '=', '₍': '(', '₎': ')',
    'ₐ': 'a', 'ₑ': 'e', 'ₕ': 'h', 'ᵢ': 'i', 'ⱼ': 'j',
    'ₖ': 'k', 'ₗ': 'l', 'ₘ': 'm', 'ₙ': 'n', 'ₒ': 'o',
    'ₚ': 'p', 'ᵣ': 'r', 'ₛ': 's', 'ₜ': 't', 'ᵤ': 'u',
    'ᵥ': 'v', 'ₓ': 'x',
}
UNICODE_SUPERSCRIPT_CHARS = ''.join(SUPERSCRIPT_CHAR_MAP.keys())
UNICODE_SUBSCRIPT_CHARS = ''.join(SUBSCRIPT_CHAR_MAP.keys())
SPECIAL_FORMULA_RUN_RE = re.compile(
    rf'(?:'
    rf'[A-Za-zΑ-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]�?[_^][A-Za-z0-9Α-Ωα-ω]+'
    rf'|'
    rf'[A-Za-zΑ-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]+[{re.escape(UNICODE_SUPERSCRIPT_CHARS + UNICODE_SUBSCRIPT_CHARS)}]+'
    rf'|'
    rf'[{MATH_OPERATOR_CHARS_ESC}]'
    rf')'
)
CAPTION_CORE_RE = re.compile(
    r'^(?:表|图|table|figure)\s*(?:\d+|[IVXLCDM]+|[一二三四五六七八九十百千万]+)?\s*[：:.-]?\s*(.+)$',
    flags=re.I,
)
BULLET_LINE_RE = re.compile(r'^[*•\-]\s+(.*)$')
ORDERED_LIST_LINE_RE = re.compile(
    r'^((?:\(?\d+[\).、])|(?:[A-Za-z][\).])|(?:[一二三四五六七八九十]+[、.]))\s*(.*)$'
)
INLINE_BULLET_SPLIT_RE = re.compile(
    r'\s+\*\s+(?=(?:对应缺口/模块|改造方案|预期收益|验证方式|技术风险|对应缺口|技术风险)[:：])'
)
PAREN_FORMULA_CANDIDATE_RE = re.compile(r'[（(][^（）()\n]{1,140}[）)]')
SCRIPTED_FORMULA_TOKEN_RE = re.compile(
    rf'(?:\\[A-Za-z]+|[A-Za-z][A-Za-z0-9]*|[Α-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]+)'
    rf'(?:\{{[^\{{}}\n]{{1,48}}\}})?'
    rf'(?:[_^](?:{FORMULA_BRACE_EXPR}|\\[A-Za-z]+|[A-Za-z0-9Α-Ωα-ω]+))+'
)
FORMULA_COMMAND_TOKEN_RE = re.compile(
    rf'\\[A-Za-z]+(?:{FORMULA_BRACE_EXPR})*(?:{FORMULA_SUBSUP_PATTERN})*'
)
FORMULA_NAME_TOKEN_PATTERN = rf'[A-Za-zΑ-Ωα-ωℒμµ{MATH_UNICODE_RANGE}][A-Za-z0-9Α-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]*'
FORMULA_SCRIPT_PART_PATTERN = rf'(?:[_^](?:{FORMULA_BRACE_EXPR}|\\[A-Za-z]+|[A-Za-z0-9Α-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]+))*'
SIMPLE_FORMULA_ATOM_PATTERN = (
    rf'(?:'
    rf'\\[A-Za-z]+(?:{FORMULA_BRACE_EXPR})*{FORMULA_SCRIPT_PART_PATTERN}'
    rf'|{FORMULA_NAME_TOKEN_PATTERN}(?:\([^()\n]{{1,80}}\))?{FORMULA_SCRIPT_PART_PATTERN}'
    rf'|[{MATH_OPERATOR_CHARS_ESC}]'
    rf'|\d+(?:\.\d+)?{FORMULA_SCRIPT_PART_PATTERN}'
    rf')'
)
INLINE_EQUATION_RUN_RE = re.compile(
    rf'(?<![A-Za-z0-9_])'
    rf'{SIMPLE_FORMULA_ATOM_PATTERN}'
    rf'(?:'
    rf'\s*(?:=|<|>|/|\+|\-|\*|·|×|÷|\\cdot|[{MATH_OPERATOR_CHARS_ESC}])\s*{SIMPLE_FORMULA_ATOM_PATTERN}'
    rf'|\s+{SIMPLE_FORMULA_ATOM_PATTERN}'
    rf')+'
    rf'(?![A-Za-z0-9_])'
)
STANDALONE_MATH_SYMBOL_RE = re.compile(
    rf'(?<![A-Za-z0-9_])(?:[{MATH_VARIABLE_CHARS}]|[Α-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]+|[{MATH_OPERATOR_CHARS_ESC}])(?![A-Za-z0-9_])'
)

LIST_ENUMERATOR_ONLY_RE = re.compile(r'^[（(]?[0-9]+[）).、]?$')
IMPLICIT_SUBSCRIPT_BASE_RE = re.compile(
    r'(?P<base>\\mathcal\{[A-Za-z]\}|\\mathbb\{[A-Za-z]\}|\\mathrm\{[A-Za-z]\}|\\[A-Za-z]+|[A-Za-z])\{(?P<sub>[A-Za-z][A-Za-z0-9]{0,15})\}'
)


def strip_outer_markdown_markers(text: str) -> str:
    value = (text or '').strip()
    patterns = [
        r'^\*\*(.*?)\*\*$',
        r'^__(.*?)__$',
        r'^`(.*?)`$',
        r'^_(.*?)_$',
    ]

    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            match = re.fullmatch(pattern, value, flags=re.S)
            if match:
                value = match.group(1).strip()
                changed = True
    return value


def normalize_compare_text(text: str) -> str:
    cleaned = strip_outer_markdown_markers(text)
    cleaned = re.sub(r'[\s`*_~：:.,，。；;、\-—–（）()\[\]{}]+', '', cleaned)
    return cleaned.lower()


def normalize_table_title_line(text: str) -> str:
    return re.sub(r'\s+', ' ', strip_outer_markdown_markers(text)).strip()


def normalize_caption_core_text(text: str) -> str:
    cleaned = normalize_table_title_line(text)
    match = CAPTION_CORE_RE.match(cleaned)
    if match:
        cleaned = match.group(1).strip()
    return normalize_compare_text(cleaned)


def is_table_title_line(text: str) -> bool:
    return bool(TABLE_TITLE_LINE_RE.match(normalize_table_title_line(text)))


def normalize_formula_spacing(text: str) -> str:
    value = (text or '').replace('\r\n', '\n').replace('\r', '\n')
    value = re.sub(r'\s*_\s*', '_', value)
    value = re.sub(r'\s*\^\s*', '^', value)
    value = re.sub(r'(\\[A-Za-z]+)\s+\{', r'\1{', value)
    return value


def collapse_spaced_math_braces(text: str) -> str:
    def _collapse(match):
        inner = match.group(1)
        tokens = inner.split()
        if len(tokens) >= 2 and all(re.fullmatch(r'[A-Za-z0-9]', tok) for tok in tokens):
            return '{' + ''.join(tokens) + '}'
        return '{' + inner.strip() + '}'

    previous = None
    value = text
    while previous != value:
        previous = value
        value = re.sub(r'\{([^{}]+)\}', _collapse, value)
    return value


def convert_unicode_scripts_to_tex(text: str) -> str:
    value = text
    if UNICODE_SUBSCRIPT_CHARS:
        sub_re = re.compile(r'[' + re.escape(UNICODE_SUBSCRIPT_CHARS) + r']+')
        value = sub_re.sub(
            lambda m: '_{' + ''.join(SUBSCRIPT_CHAR_MAP[ch] for ch in m.group(0)) + '}',
            value,
        )
    if UNICODE_SUPERSCRIPT_CHARS:
        sup_re = re.compile(r'[' + re.escape(UNICODE_SUPERSCRIPT_CHARS) + r']+')
        value = sup_re.sub(
            lambda m: '^{' + ''.join(SUPERSCRIPT_CHAR_MAP[ch] for ch in m.group(0)) + '}',
            value,
        )
    return value


def repair_broken_formula_glyphs(text: str) -> str:
    value = text or ''
    value = value.replace('￾', '').replace('￼', '').replace('\u00ad', '')
    value = re.sub(
        r'([xX])�(?=_[dDtT])',
        lambda m: r'\tilde{' + m.group(1).lower() + '}',
        value,
    )
    value = re.sub(
        r'([A-Za-zΑ-Ωα-ω])�(?=_[A-Za-z0-9])',
        lambda m: r'\hat{' + m.group(1) + '}',
        value,
    )
    value = re.sub(
        r'([A-Za-zΑ-Ωα-ω])�',
        lambda m: r'\hat{' + m.group(1) + '}',
        value,
    )
    return value



def split_joined_latex_commands(text: str) -> str:
    value = text or ''
    return LATEX_COMMAND_SPLIT_RE.sub(r'\1 ', value)


def normalize_broken_formula_plain_text(text: str) -> str:
    value = text or ''
    if not value:
        return ''

    value = value.replace('$$', ' ')
    value = re.sub(r'\([A-Za-z]+)\$', lambda m: '\\' + m.group(1) + ' ', value)
    value = re.sub(r'\$\([A-Za-z]+)', lambda m: ' \\' + m.group(1), value)
    value = re.sub(
        r'([A-Za-z0-9Α-Ωα-ωℒμµ{}\]\)])\$(?=[A-Za-z0-9Α-Ωα-ωℒμµ\{\[(])',
        lambda m: m.group(1) + ' ',
        value,
    )
    value = re.sub(r'(?<=[\A-Za-z0-9Α-Ωα-ωℒμµ{}\]\)])\$(?=\s|[，。；：,.;:）\]\)])', ' ', value)
    value = split_joined_latex_commands(value)
    value = re.sub(r'\s{2,}', ' ', value)
    return value


def normalize_math_unicode_to_latex(text: str) -> str:
    value = repair_broken_formula_glyphs(text)
    value = split_joined_latex_commands(value)
    value = value.replace('（', '(').replace('）', ')').replace('，', ',').replace('：', ':')
    value = convert_unicode_scripts_to_tex(value)
    for raw, latex in SPECIAL_FORMULA_CHAR_MAP.items():
        value = value.replace(raw, latex)

    value = unicodedata.normalize('NFKD', value)

    accent_map = {
        '\u0302': r'\hat',
        '\u0303': r'\tilde',
        '\u0304': r'\bar',
        '\u0307': r'\dot',
    }
    for mark, cmd in accent_map.items():
        value = re.sub(
            rf'(\\[A-Za-z]+|[A-Za-zΑ-Ωα-ω]){mark}',
            lambda m, cmd=cmd: cmd + '{' + m.group(1) + '}',
            value,
        )

    value = re.sub(r'[\u0300-\u036F]+', '', value)

    for raw, latex in GREEK_LATEX_MAP.items():
        value = value.replace(raw, latex)
    return value


def is_list_enumerator_text(text: str) -> bool:
    candidate = extract_formula_text(text)
    if not candidate:
        return False
    candidate = candidate.strip().replace('（', '(').replace('）', ')')
    return bool(LIST_ENUMERATOR_ONLY_RE.fullmatch(candidate)) or bool(re.fullmatch(r'^[A-Za-z][\).]$', candidate))


def repair_implicit_subscripts(expr: str) -> str:
    allowed_symbol_cmds = {
        r'\alpha', r'\beta', r'\gamma', r'\delta', r'\epsilon', r'\theta',
        r'\lambda', r'\mu', r'\sigma', r'\omega', r'\phi', r'\psi',
        r'\rho', r'\nu', r'\kappa', r'\xi', r'\Delta', r'\Gamma',
        r'\Lambda', r'\Sigma', r'\Pi', r'\Omega', r'\Phi', r'\Psi',
    }

    def _replace(match):
        base = match.group('base')
        sub = match.group('sub')
        if base.startswith(r'\mathcal{') or base.startswith(r'\mathbb{') or base.startswith(r'\mathrm{'):
            return f'{base}_{{{sub}}}'
        if len(base) == 1 and base.isalpha():
            return f'{base}_{{{sub}}}'
        if base in allowed_symbol_cmds:
            return f'{base}_{{{sub}}}'
        return match.group(0)

    previous = None
    value = expr
    while previous != value:
        previous = value
        value = IMPLICIT_SUBSCRIPT_BASE_RE.sub(_replace, value)
    return value


def formula_has_strong_signal(text: str) -> bool:
    value = text or ''
    if not value.strip():
        return False
    if any(ch in value for ch in MATH_OPERATOR_CHARS):
        return True
    if any(ch in value for ch in UNICODE_SUPERSCRIPT_CHARS + UNICODE_SUBSCRIPT_CHARS):
        return True
    if re.search(r'[=<>^_+*/\\]|\bcdot\b|\\[A-Za-z]+', value):
        return True
    if re.search(r'[Α-Ωα-ωℒμµ]', value):
        return True
    return False


def normalize_formula_script_groups(expr: str) -> str:
    r"""修复 L_aug_i、L_{aug}_i、L_{\mathrm{aug}_i} 这类会导致 KaTeX/MathText 报错的下标。"""
    value = expr or ''

    script_body_pattern = r'(?:[^{}]|\\[A-Za-z]+\{[^{}]*\})+'

    def _strip_outer_script_braces(script: str) -> str:
        script = (script or '').strip()
        if script.startswith('{') and script.endswith('}'):
            return script[1:-1].strip()
        return script

    def _merge_script_parts(op: str, left: str, right: str) -> str:
        left = _strip_outer_script_braces(left)
        right = _strip_outer_script_braces(right)
        return f'{op}{{{left},{right}}}'

    # 合并连续的下标/上标，覆盖 z_{drug}_{specific}、L_{aug}_i、L_i_{aug}。
    previous = None
    while previous != value:
        previous = value
        value = re.sub(
            rf'([_^])\{{({script_body_pattern})\}}\s*\1\{{({script_body_pattern})\}}',
            lambda m: _merge_script_parts(m.group(1), m.group(2), m.group(3)),
            value,
        )
        value = re.sub(
            rf'([_^])\{{({script_body_pattern})\}}\s*\1([A-Za-z0-9Α-Ωα-ω]+|\\[A-Za-z]+)',
            lambda m: _merge_script_parts(m.group(1), m.group(2), m.group(3)),
            value,
        )
        value = re.sub(
            rf'([_^])([A-Za-z0-9Α-Ωα-ω]+|\\[A-Za-z]+)\s*\1\{{({script_body_pattern})\}}',
            lambda m: _merge_script_parts(m.group(1), m.group(2), m.group(3)),
            value,
        )

    def _flatten_mathrm_scripts(text: str) -> str:
        previous_inner = None
        current = text
        while previous_inner != current:
            previous_inner = current
            current = re.sub(
                r'\\mathrm\{([^{}]*)\}\s*_\s*\{([^{}]+)\}',
                lambda m: r'\mathrm{' + m.group(1).replace(r'\_', ',') + ',' + m.group(2).strip() + '}',
                current,
            )
            current = re.sub(
                r'\\mathrm\{([^{}]*)\}\s*_\s*([A-Za-z0-9Α-Ωα-ω]+)',
                lambda m: r'\mathrm{' + m.group(1).replace(r'\_', ',') + ',' + m.group(2).strip() + '}',
                current,
            )
            current = re.sub(
                r'\\mathrm\{([^{}]*?)_([A-Za-z0-9Α-Ωα-ω]+)\}',
                lambda m: r'\mathrm{' + m.group(1).replace(r'\_', ',') + ',' + m.group(2).strip() + '}',
                current,
            )
        return current

    value = _flatten_mathrm_scripts(value)

    def _normalize_script(match):
        op = match.group(1)
        body = _flatten_mathrm_scripts((match.group(2) or '').strip())
        if not body:
            return match.group(0)
        if re.fullmatch(r'\\mathrm\{[^{}]*\}', body):
            return f'{op}{{{body}}}'
        if body.startswith('\\') and not re.fullmatch(r'\\[A-Za-z]+', body):
            return f'{op}{{{body}}}'
        if re.search(r'[+\-*/=<>]', body):
            return f'{op}{{{body}}}'
        if re.search(r'[A-Za-z]{2,}|_', body):
            safe = body.replace('\\', r'\backslash ')
            safe = safe.replace('_', ',')
            safe = re.sub(r'\s+', r'\\ ', safe)
            return f'{op}{{\\mathrm{{{safe}}}}}'
        return f'{op}{{{body}}}'

    value = re.sub(rf'([_^])\{{({script_body_pattern})\}}', _normalize_script, value)
    value = _flatten_mathrm_scripts(value)

    # 最后兜底：把 L_{\mathrm{aug}}_i 合并成 L_{\mathrm{aug,i}}，避免 Double subscript。
    previous = None
    while previous != value:
        previous = value
        value = re.sub(
            r'([A-Za-zΑ-Ωα-ω])_\{\\mathrm\{([^{}]+)\}\}\s*_\s*\{?([A-Za-z0-9Α-Ωα-ω]+)\}?',
            lambda m: f'{m.group(1)}_{{\\mathrm{{{m.group(2)},{m.group(3)}}}}}',
            value,
        )
    return value

def explode_inline_numbered_segments(text: str) -> List[str]:
    value = (text or '').strip()
    if not value:
        return []

    colon_match = re.search(r'[：:]', value)
    if not colon_match:
        return [value]

    prefix = value[:colon_match.end()].strip()
    suffix = value[colon_match.end():].strip()
    numbered_parts = [
        part.strip(' ；;')
        for part in re.split(r'(?=(?:\(?\d+[\).、]))', suffix)
        if part and part.strip(' ；;')
    ]
    if len(numbered_parts) < 2:
        return [value]

    return [prefix] + numbered_parts

def append_list_item_blocks(blocks: List[Tuple[str, object]], item_text: str, prefix: str = '- '):
    pieces = explode_inline_numbered_segments(item_text)
    if not pieces:
        return
    for idx, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        if idx == 0 and prefix:
            blocks.append(("list_item", f"{prefix}{piece}"))
        elif re.match(r'^\(?\d+[\).、]', piece):
            blocks.append(("list_item", piece))
        else:
            blocks.append(("list_item", f"- {piece}"))

def looks_like_formula_text(text: str) -> bool:
    candidate = extract_formula_text(text)
    if not candidate:
        return False
    if is_list_enumerator_text(candidate):
        return False

    normalized = collapse_spaced_math_braces(normalize_formula_spacing(candidate))
    normalized = normalize_math_unicode_to_latex(normalized)
    normalized = repair_implicit_subscripts(normalized)

    if (
        AUTO_FORMULA_RUN_RE.fullmatch(normalized)
        or SPECIAL_FORMULA_RUN_RE.fullmatch(normalized)
        or SCRIPTED_FORMULA_TOKEN_RE.fullmatch(normalized)
    ):
        return True

    if FORMULA_COMMAND_TOKEN_RE.fullmatch(normalized):
        return True

    repaired = repair_broken_formula_glyphs(normalized)
    if repaired != normalized and (AUTO_FORMULA_RUN_RE.search(repaired) or '_' in repaired or '^' in repaired):
        return True

    if SCRIPTED_FORMULA_TOKEN_RE.search(normalized) or FORMULA_COMMAND_TOKEN_RE.search(normalized):
        return True

    explicit_tokens = [
        r'\frac', r'\sum', r'\prod', r'\sqrt', r'\alpha', r'\beta', r'\gamma',
        r'\theta', r'\lambda', r'\mu', r'\sigma', r'\omega', r'\mathbf',
        r'\mathrm', r'\mathcal', r'\mathbb', r'\hat', r'\tilde', r'\left',
        r'\right', r'\begin', r'\end', r'\log', r'\exp', r'\arg', r'\max',
        r'\min',
    ]
    if any(token in normalized for token in explicit_tokens):
        return True

    if re.search(r'[Α-Ωα-ωℒμµ]', normalized):
        return True

    if any(ch in normalized for ch in UNICODE_SUPERSCRIPT_CHARS + UNICODE_SUBSCRIPT_CHARS):
        return True

    if any(ch in normalized for ch in MATH_OPERATOR_CHARS):
        return True

    has_letter = bool(re.search(r'[A-Za-zΑ-Ωα-ω]', normalized))
    operator_count = sum(ch in normalized for ch in "=<>^_+-*/\\")
    brace_count = normalized.count('{') + normalized.count('}')

    if has_letter and ('_' in normalized or '^' in normalized):
        return True
    if has_letter and brace_count >= 2 and operator_count >= 1:
        return True
    return has_letter and operator_count >= 2

def extract_formula_text(text: str) -> str:
    candidate = (text or '').strip()
    if not candidate:
        return ''

    if candidate.startswith('$$') and candidate.endswith('$$') and len(candidate) >= 4:
        candidate = candidate[2:-2]
    elif candidate.startswith(r'\(') and candidate.endswith(r'\)') and len(candidate) >= 4:
        candidate = candidate[2:-2]
    elif candidate.startswith(r'\[') and candidate.endswith(r'\]') and len(candidate) >= 4:
        candidate = candidate[2:-2]
    elif candidate.startswith('`') and candidate.endswith('`') and len(candidate) >= 2:
        candidate = candidate[1:-1]

    candidate = re.sub(r'^```(?:math|latex|tex)?\s*', '', candidate, flags=re.I)
    candidate = re.sub(r'\s*```$', '', candidate)
    candidate = candidate.replace('\r\n', '\n').replace('\r', '\n')
    candidate = '\n'.join([line.strip() for line in candidate.split('\n') if line.strip()])
    return candidate.strip()


def sanitize_formula_for_render(text: str) -> str:
    expr = extract_formula_text(text)
    if not expr:
        return ''

    expr = normalize_broken_formula_plain_text(expr)
    expr = repair_broken_formula_glyphs(expr)
    expr = split_joined_latex_commands(expr)
    expr = expr.replace('$', ' ')
    expr = normalize_formula_spacing(expr)
    expr = collapse_spaced_math_braces(expr)
    expr = normalize_math_unicode_to_latex(expr)
    expr = repair_implicit_subscripts(expr)

    expr = re.sub(r'\\begin\{[^{}]+\}', '', expr)
    expr = re.sub(r'\\end\{[^{}]+\}', '', expr)
    expr = expr.replace('&', ' ')
    expr = expr.replace('\\\\', ' ')
    expr = expr.replace(r'\displaystyle', '')
    expr = expr.replace('−', '-').replace('–', '-').replace('—', '-')
    expr = re.sub(r'\\label\{.*?\}', '', expr)
    expr = re.sub(r'\\nonumber\b', '', expr)
    expr = re.sub(r'(?<=\d)\s+(?=\d)', '', expr)
    expr = re.sub(
        r'\\text\s*\{([^{}]*)\}',
        lambda m: r'\mathrm{' + m.group(1).replace(' ', r'\ ') + '}',
        expr,
    )
    expr = re.sub(
        r'\\operatorname\s*\{([^{}]*)\}',
        lambda m: r'\mathrm{' + m.group(1).replace(' ', r'\ ') + '}',
        expr,
    )
    expr = re.sub(
        r'\\mathrm\{([^{}]*)\}',
        lambda m: r'\mathrm{' + ''.join(m.group(1).split()) + '}',
        expr,
    )
    expr = re.sub(r'_(?!\{)\s*(\\[A-Za-z]+)', r'_{\1}', expr)
    expr = re.sub(r'\^(?!\{)\s*(\\[A-Za-z]+)', r'^{\1}', expr)
    expr = re.sub(r'_(?!\{)\s*([A-Za-z0-9]{2,})', r'_{\1}', expr)
    expr = re.sub(r'\^(?!\{)\s*([A-Za-z0-9]{2,})', r'^{\1}', expr)
    expr = normalize_formula_script_groups(expr)
    expr = re.sub(r'\\Sigma(?=\s*[({])', r'\\sum', expr)
    expr = expr.replace('...', r'\ldots')
    expr = re.sub(r'\s*\*\s*', lambda m: r' \cdot ', expr)
    expr = re.sub(r'\s+', ' ', expr).strip()
    return expr


def should_auto_render_formula(candidate: str) -> bool:
    stripped = extract_formula_text(candidate)
    if not stripped:
        return False
    if is_list_enumerator_text(stripped):
        return False

    normalized = collapse_spaced_math_braces(normalize_formula_spacing(stripped))
    normalized_latex = normalize_math_unicode_to_latex(normalized)

    if INLINE_EQUATION_RUN_RE.fullmatch(normalized) and formula_has_strong_signal(normalized):
        return True
    if SCRIPTED_FORMULA_TOKEN_RE.search(normalized) or FORMULA_COMMAND_TOKEN_RE.search(normalized):
        return True
    if any(ch in normalized for ch in MATH_OPERATOR_CHARS):
        return True
    if re.fullmatch(rf'[Α-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]+', normalized):
        return True
    if re.fullmatch(rf'[{MATH_VARIABLE_CHARS}]', normalized):
        return True

    if '�' in stripped or any(ch in stripped for ch in UNICODE_SUPERSCRIPT_CHARS + UNICODE_SUBSCRIPT_CHARS):
        return True
    if re.fullmatch(r'[A-Z]{2,}(?:-[A-Z]{2,})*', stripped):
        return False
    if re.fullmatch(r'[A-Za-z]{2,}', stripped):
        return False
    return looks_like_formula_text(normalized_latex)


def has_inline_math_context(working: str, start: int, end: int) -> bool:
    token = working[start:end].strip()
    if not token:
        return False
    if any(ch in token for ch in MATH_OPERATOR_CHARS):
        return True
    if re.fullmatch(rf'[Α-Ωα-ωℒμµ{MATH_UNICODE_RANGE}]+', token):
        return True

    around = working[max(0, start - 18):min(len(working), end + 18)]
    if re.search(r'[=+\-*/<>_^{}]|\\cdot|' + rf'[{MATH_OPERATOR_CHARS_ESC}]', around):
        return True
    if re.search(r'[\u4e00-\u9fff]', around):
        return True
    return False


def find_standalone_math_symbol_match(working: str, start_pos: int):
    match = STANDALONE_MATH_SYMBOL_RE.search(working, start_pos)
    while match:
        if has_inline_math_context(working, match.start(), match.end()):
            return match
        match = STANDALONE_MATH_SYMBOL_RE.search(working, match.start() + 1)
    return None


def find_parenthetical_formula_match(working: str, start_pos: int):
    match = PAREN_FORMULA_CANDIDATE_RE.search(working, start_pos)
    while match:
        candidate = match.group(0).strip()
        inner = candidate[1:-1].strip()
        if should_auto_render_formula(inner):
            return match
        match = PAREN_FORMULA_CANDIDATE_RE.search(working, match.start() + 1)
    return None


def collect_formula_candidate_matches(working: str, cursor: int):
    candidate_matches = []
    for pattern in (
        INLINE_EQUATION_RUN_RE,
        SCRIPTED_FORMULA_TOKEN_RE,
        AUTO_FORMULA_RUN_RE,
        SPECIAL_FORMULA_RUN_RE,
    ):
        match = pattern.search(working, cursor)
        if match:
            candidate_matches.append(match)

    standalone_match = find_standalone_math_symbol_match(working, cursor)
    if standalone_match:
        candidate_matches.append(standalone_match)

    paren_match = find_parenthetical_formula_match(working, cursor)
    if paren_match:
        candidate_matches.append(paren_match)
    return candidate_matches


def formula_inline_markdown(formula_text: str, display: bool = False) -> str:
    expr = sanitize_formula_for_render(formula_text)
    if not expr:
        expr = extract_formula_text(formula_text)
    expr = (expr or '').strip()
    if not expr:
        return (formula_text or '').strip()

    if display or '\n' in expr:
        return f"$$\n{expr}\n$$"
    return f"${expr}$"


def wrap_plain_text_for_markdown(text: str) -> str:
    if not text:
        return ''

    working = normalize_broken_formula_plain_text(text)
    working = collapse_spaced_math_braces(normalize_formula_spacing(working))
    working = split_joined_latex_commands(working)
    working = working.replace('$', ' ')
    parts: List[str] = []
    cursor = 0

    while cursor < len(working):
        candidate_matches = collect_formula_candidate_matches(working, cursor)
        if not candidate_matches:
            parts.append(working[cursor:])
            break

        match = min(candidate_matches, key=lambda m: (m.start(), -(m.end() - m.start())))
        start, end = match.span()
        candidate = match.group(0).strip()

        if start > cursor:
            parts.append(working[cursor:start])

        if (
            not is_list_enumerator_text(candidate)
            and (
                should_auto_render_formula(candidate)
                or '�' in candidate
                or any(ch in candidate for ch in UNICODE_SUPERSCRIPT_CHARS + UNICODE_SUBSCRIPT_CHARS)
                or candidate[:1] in {'(', '（'}
            )
        ):
            parts.append(formula_inline_markdown(candidate, display=False))
        else:
            parts.append(candidate)

        cursor = end

    return ''.join(parts)


def convert_inline_formula_markup_to_markdown(text: str) -> str:
    value = text or ''
    if not value:
        return ''

    parts: List[str] = []
    start = 0
    for match in INLINE_MARKUP_TOKEN_RE.finditer(value):
        if match.start() > start:
            parts.append(wrap_plain_text_for_markdown(value[start:match.start()]))

        token = match.group(0)
        if token.startswith('**') and token.endswith('**'):
            parts.append(f"**{convert_inline_formula_markup_to_markdown(token[2:-2])}**")
        elif token.startswith('__') and token.endswith('__'):
            parts.append(f"__{convert_inline_formula_markup_to_markdown(token[2:-2])}__")
        elif token.startswith('$') and token.endswith('$'):
            parts.append(formula_inline_markdown(token[1:-1], display=False))
        elif token.startswith(r'\(') and token.endswith(r'\)'):
            parts.append(formula_inline_markdown(token[2:-2], display=False))
        elif token.startswith(r'\[') and token.endswith(r'\]'):
            parts.append(formula_inline_markdown(token[2:-2], display=True))
        elif token.startswith('`') and token.endswith('`'):
            inner = token[1:-1]
            if should_auto_render_formula(inner):
                parts.append(formula_inline_markdown(inner, display=False))
            else:
                parts.append(token)
        start = match.end()

    if start < len(value):
        parts.append(wrap_plain_text_for_markdown(value[start:]))

    return ''.join(parts)


# ==========================================
# 模块 10：Markdown 自定义块解析
# ==========================================
def split_markdown_blocks(md_text: str) -> List[Tuple[str, object]]:
    """
    自定义 Markdown 块解析器。

    只解析当前系统真正会生成的几类结构：
    - 标题：## / ###
    - 图片：![caption](id)
    - 表格标题：表1：... / Table 1: ...
    - 公式块：$$...$$ / \\[...\\] / ```latex``` / 常见公式环境
    - Markdown 表格：| a | b |
    - 列表项：* xxx / 1) xxx
    - 普通段落：空行分隔
    """
    text = normalize_report_markdown(md_text)
    lines = text.split("\n")
    blocks: List[Tuple[str, object]] = []

    paragraph_buffer: List[str] = []

    def flush_paragraph():
        if paragraph_buffer:
            joined = " ".join([x.strip() for x in paragraph_buffer if x.strip()]).strip()
            if joined:
                fragments = [frag.strip() for frag in INLINE_BULLET_SPLIT_RE.split(joined) if frag and frag.strip()]
                if len(fragments) > 1:
                    head = fragments[0]
                    if head:
                        blocks.append(("paragraph", head))
                    for frag in fragments[1:]:
                        cleaned = frag.lstrip("*•- ").strip()
                        append_list_item_blocks(blocks, cleaned, prefix='- ')
                else:
                    blocks.append(("paragraph", joined))
            paragraph_buffer.clear()

    def is_structural_block(line: str) -> bool:
        if not line:
            return True
        if line.startswith("## ") or line.startswith("### "):
            return True
        if line.startswith("```"):
            return True
        if re.match(r'^\\begin\{[^{}]+\}$', line):
            return True
        if line.startswith("$$") or line.startswith(r"\["):
            return True
        if re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line):
            return True
        if is_table_title_line(line):
            return True
        if line.startswith("|"):
            return True
        if BULLET_LINE_RE.match(line) or ORDERED_LIST_LINE_RE.match(line):
            return True
        return False

    def consume_list_item(start_index: int, initial_text: str) -> Tuple[str, int]:
        item_lines = [initial_text.strip()]
        j = start_index + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if not nxt or is_structural_block(nxt):
                break
            item_lines.append(nxt)
            j += 1
        return " ".join([x for x in item_lines if x]).strip(), j

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            blocks.append(("h2", stripped[3:].strip()))
            i += 1
            continue

        if stripped.startswith("### "):
            flush_paragraph()
            blocks.append(("h3", stripped[4:].strip()))
            i += 1
            continue

        bullet_match = BULLET_LINE_RE.match(stripped)
        if bullet_match:
            flush_paragraph()
            item_text, next_i = consume_list_item(i, bullet_match.group(1))
            append_list_item_blocks(blocks, item_text, prefix='- ')
            i = next_i
            continue

        ordered_match = ORDERED_LIST_LINE_RE.match(stripped)
        if ordered_match:
            flush_paragraph()
            marker = ordered_match.group(1).strip()
            item_text, next_i = consume_list_item(i, ordered_match.group(2))
            pieces = explode_inline_numbered_segments(item_text)
            if pieces:
                first = f"{marker} {pieces[0]}".strip()
                blocks.append(("list_item", first))
                for piece in pieces[1:]:
                    piece = piece.strip()
                    if not piece:
                        continue
                    if re.match(r'^\(?\d+[\).、]', piece):
                        blocks.append(("list_item", piece))
                    else:
                        blocks.append(("list_item", f"- {piece}"))
            i = next_i
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            fence_lang = stripped[3:].strip().lower()
            fenced_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                fenced_lines.append(lines[i].rstrip())
                i += 1
            if i < len(lines) and lines[i].strip().startswith("```"):
                i += 1

            block_text = "\n".join([x for x in fenced_lines if x.strip()]).strip()
            if block_text:
                if fence_lang in {"math", "latex", "tex"} or looks_like_formula_text(block_text):
                    blocks.append(("math_block", block_text))
                else:
                    blocks.append(("code_block", (fence_lang or "text", "\n".join(fenced_lines).strip())))
            continue

        if re.match(r'^\\begin\{[^{}]+\}$', stripped):
            flush_paragraph()
            env_lines = [stripped]
            i += 1
            while i < len(lines):
                env_lines.append(lines[i].strip())
                if re.match(r'^\\end\{[^{}]+\}$', lines[i].strip()):
                    i += 1
                    break
                i += 1
            blocks.append(("math_block", "\n".join(env_lines)))
            continue

        if stripped.startswith("$$") or stripped.startswith(r"\["):
            flush_paragraph()
            opener, closer = ("$$", "$$") if stripped.startswith("$$") else (r"\[", r"\]")
            formula_lines = []
            current = stripped

            if current.startswith(opener):
                current = current[len(opener):]

            if current.endswith(closer) and current.strip() != closer:
                current = current[:-len(closer)]
                if current.strip():
                    formula_lines.append(current.strip())
                i += 1
            else:
                if current.strip():
                    formula_lines.append(current.strip())
                i += 1
                while i < len(lines):
                    current = lines[i].strip()
                    if current.endswith(closer):
                        tail = current[:-len(closer)].strip()
                        if tail:
                            formula_lines.append(tail)
                        i += 1
                        break
                    formula_lines.append(current)
                    i += 1

            formula_text = "\n".join([x for x in formula_lines if x])
            if formula_text:
                blocks.append(("math_block", formula_text))
            continue

        img_match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', stripped)
        if img_match:
            flush_paragraph()
            blocks.append(("image", (img_match.group(1).strip(), img_match.group(2).strip())))
            i += 1
            continue

        if is_table_title_line(stripped):
            flush_paragraph()
            blocks.append(("table_title", normalize_table_title_line(stripped)))
            i += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            table_lines = [stripped]
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("|"):
                table_lines.append(lines[j].strip())
                j += 1
            blocks.append(("md_table", table_lines))
            i = j
            continue

        paragraph_buffer.append(stripped)
        i += 1

    flush_paragraph()
    return blocks


def split_title_and_body(md_text: str) -> Tuple[str, str]:
    """从完整 markdown 中拆出文档主标题和正文。"""
    text = normalize_report_markdown(md_text)
    lines = text.split("\n")
    if lines and lines[0].startswith("# "):
        return lines[0][2:].strip(), "\n".join(lines[1:]).strip()
    return "论文全维度深度透视报告", text


CURRENT_REPORT_SECTION_TITLES = [
    "研究问题与核心贡献",
    "背景、研究缺口与前人路线",
    "方法总览与整体数据流",
    "关键模块逐层机制剖析",
    "实验设计、关键证据与论点验证",
    "局限性与未解决问题",
    "同问题方法比较与综合分析",
    "面向后续研究的可执行创新路线",
]
REMOVED_REPORT_SECTION_TITLES = {"复现要点与方法适用边界"}


def canonicalize_report_section_core(title: str) -> str:
    raw = (title or "").strip()
    raw = re.sub(r'^#+\s*', '', raw)
    raw = re.sub(r'^\d+\s*[.．、:：\-]?\s*', '', raw)
    normalized = normalize_compare_text(raw)

    alias_map = {
        "同问题方法比较与综合分析": [
            "同问题方法比较与综合分析",
            "同问题论文比较与综合分析",
            "同问题比较与综合分析",
            "相关方法比较与综合分析",
            "历史论文比较与综合分析",
        ]
    }

    for canonical_title, aliases in alias_map.items():
        if any(normalize_compare_text(alias) == normalized for alias in aliases):
            return canonical_title

    for candidate in CURRENT_REPORT_SECTION_TITLES + list(REMOVED_REPORT_SECTION_TITLES):
        if normalize_compare_text(candidate) == normalized:
            return candidate
    return raw



def prepare_report_markdown_for_display(
    md_text: str,
    images_dict: Optional[Dict[str, str]] = None,
    vision_summaries: str = '',
) -> str:
    """
    前端展示前也执行一次完整报告后处理：
    1. 修正正文中的图表引用
    2. 同步最终图片 caption 编号
    3. 兼容旧缓存/旧历史报告中“正文编号没跟着改”的情况
    4. 最后再做块级与公式序列化，保证页面渲染稳定
    """
    normalized_sections = normalize_report_markdown(md_text)
    image_ids = list((images_dict or {}).keys())

    try:
        normalized_sections = postprocess_generated_report_markdown(
            normalized_sections,
            image_ids=image_ids,
            vision_summaries=vision_summaries or '',
        )
    except Exception:
        # 展示层兜底：后处理失败时至少不阻塞页面显示
        pass

    try:
        doc_title, body = split_title_and_body(normalized_sections)
        blocks = split_markdown_blocks(body)
        return serialize_report_blocks(blocks, doc_title)
    except Exception:
        return normalized_sections

def convert_inline_formulas_in_table_line(line: str) -> str:
    stripped = (line or '').strip()
    if not stripped.startswith('|'):
        return line

    cells = [cell.strip() for cell in stripped.strip('|').split('|')]
    if cells and all(re.fullmatch(r'[:\- ]+', cell) for cell in cells):
        return stripped

    rendered_cells = [convert_inline_formula_markup_to_markdown(cell) for cell in cells]
    return '| ' + ' | '.join(rendered_cells) + ' |'


def serialize_report_blocks(blocks: List[Tuple[str, object]], doc_title: str) -> str:
    lines: List[str] = [f"# {doc_title}"]

    for block_type, payload in blocks:
        if block_type == 'h2':
            lines.extend(['', f"## {convert_inline_formula_markup_to_markdown(str(payload))}"])
        elif block_type == 'h3':
            lines.extend(['', f"### {convert_inline_formula_markup_to_markdown(str(payload))}"])
        elif block_type == 'table_title':
            lines.extend(['', convert_inline_formula_markup_to_markdown(normalize_table_title_line(str(payload)))])
        elif block_type == 'paragraph':
            lines.extend(['', convert_inline_formula_markup_to_markdown(str(payload))])
        elif block_type == 'list_item':
            lines.extend(['', convert_inline_formula_markup_to_markdown(str(payload))])
        elif block_type == 'math_block':
            lines.extend(['', formula_inline_markdown(str(payload), display=True)])
        elif block_type == 'code_block':
            lang, code = payload
            lines.extend(['', f"```{lang}", str(code), "```"])
        elif block_type == 'image':
            caption, key = payload
            rendered_caption = convert_inline_formula_markup_to_markdown(str(caption)) if caption else ''
            lines.extend(['', f"![{rendered_caption}]({key})"])
        elif block_type == 'md_table':
            lines.append('')
            lines.extend([convert_inline_formulas_in_table_line(str(row)) for row in payload])
        else:
            lines.extend(['', str(payload)])

    return normalize_report_markdown('\n'.join(lines))


def normalize_report_image_key(image_key: str) -> str:
    value = (image_key or '').strip().replace('\\', '/')
    value = value.split('#', 1)[0].split('?', 1)[0]
    value = value.split('/')[-1]
    return re.sub(r'\s+', '', value).lower()


def deduplicate_report_image_blocks(blocks: List[Tuple[str, object]]) -> List[Tuple[str, object]]:
    deduped: List[Tuple[str, object]] = []
    seen_image_keys = set()

    for block_type, payload in blocks:
        if block_type != 'image':
            deduped.append((block_type, payload))
            continue

        caption, key = payload
        dedup_key = normalize_report_image_key(str(key))
        if dedup_key and dedup_key in seen_image_keys:
            if deduped and deduped[-1][0] == 'table_title':
                last_title = str(deduped[-1][1])
                caption_text = (caption or '').strip()
                if not caption_text or captions_equivalent(caption_text, last_title):
                    deduped.pop()
            continue

        if dedup_key:
            seen_image_keys.add(dedup_key)
        deduped.append(('image', ((caption or '').strip(), key)))

    return deduped


REPORT_IMAGE_LINE_RE = re.compile(r'^!\[(?P<caption>.*?)\]\((?P<key>.*?)\)\s*$')
REPORT_FIG_TABLE_LABEL_RE = re.compile(r'(图|表)\s*([0-9一二三四五六七八九十百千万]+)')
REPORT_EN_LABEL_RE = re.compile(r'\b(Figure|Fig\.?|Table)\s*([0-9]+)\b', flags=re.I)
REPORT_IMAGE_ID_LIKE_RE = re.compile(
    r'(?:image[_\-]?[0-9a-f]{6,}|fig(?:ure)?[_\-]?\d+|table[_\-]?\d+|[0-9a-f]{16,}|[\w\-.]+\.(?:png|jpg|jpeg|webp))',
    flags=re.I,
)


def normalize_report_label(kind: str, number: str) -> str:
    return f"{(kind or '图').strip()}{re.sub(r'\\s+', '', str(number or '').strip())}"


def extract_report_label(text: str) -> Optional[Tuple[str, str]]:
    value = text or ''
    match = REPORT_FIG_TABLE_LABEL_RE.search(value)
    if match:
        return match.group(1), re.sub(r'\s+', '', match.group(2))
    match = REPORT_EN_LABEL_RE.search(value)
    if match:
        kind = '表' if match.group(1).lower().startswith('table') else '图'
        return kind, match.group(2)
    return None


def is_table_like_figure_text(text: str) -> bool:
    value = (text or '').strip()
    lower = value.lower()
    table_keywords = [
        'table', 'tabular', '表格', '数据表', '结果表', '对比表', '消融表', '指标表', '性能表', '统计表'
    ]
    figure_keywords = ['figure', 'fig.', '架构图', '流程图', '曲线图', '示意图', '模块图', '框图', '散点图', '折线图']
    if any(keyword in lower for keyword in ['table', 'tabular']) or any(keyword in value for keyword in table_keywords[2:]):
        return True
    if any(keyword in lower for keyword in ['figure', 'fig.']) or any(keyword in value for keyword in figure_keywords[2:]):
        return False
    label = extract_report_label(value)
    return bool(label and label[0] == '表')


def parse_vision_figure_metadata(vision_summaries: str) -> Dict[str, Dict[str, str]]:
    metadata: Dict[str, Dict[str, str]] = {}
    text = vision_summaries or ''
    if not text.strip():
        return metadata

    chunks = re.split(r'\n\s*---\s*图表标识\s*:\s*(.*?)\s*---\s*\n', text)
    pairs: List[Tuple[str, str]] = []
    if len(chunks) >= 3:
        for idx in range(1, len(chunks), 2):
            pairs.append((chunks[idx].strip(), chunks[idx + 1] if idx + 1 < len(chunks) else ''))
    else:
        card_chunks = re.split(r'(?=<FIGURE_CARD>)', text)
        pairs = [('', chunk) for chunk in card_chunks if chunk.strip()]

    for fallback_key, chunk in pairs:
        key = fallback_key
        id_match = re.search(r'图像ID\s*[：:]\s*(.+)', chunk)
        if id_match:
            candidate = id_match.group(1).strip().splitlines()[0].strip()
            if candidate and candidate not in {'未显示', '无'}:
                key = candidate
        if not key:
            continue

        type_match = re.search(r'图表类型\s*[：:]\s*(.+)', chunk)
        type_text = type_match.group(1).strip().splitlines()[0].strip() if type_match else ''
        original_match = re.search(r'原文编号\s*[：:]\s*(.+)', chunk)
        original_text = original_match.group(1).strip().splitlines()[0].strip() if original_match else ''
        caption_match = re.search(r'推荐图注\s*[：:]\s*(.+)', chunk)
        caption = caption_match.group(1).strip().splitlines()[0].strip() if caption_match else ''

        label = extract_report_label(original_text) or extract_report_label(caption)
        kind = '表' if is_table_like_figure_text(type_text) or is_table_like_figure_text(original_text) or (label and label[0] == '表') else '图'
        number = label[1] if label else ''

        metadata[normalize_report_image_key(key)] = {
            'key': key,
            'kind': kind,
            'number': number,
            'caption': caption,
            'type_text': type_text,
            'original_text': original_text,
        }
    return metadata


def strip_label_prefix(text: str) -> str:
    value = (text or '').strip()
    value = re.sub(r'^(?:图|表)\s*[0-9一二三四五六七八九十百千万]+\s*[：:：.．\-—–]?\s*', '', value)
    value = re.sub(r'^(?:Figure|Fig\.?|Table)\s*[0-9]+\s*[：:：.．\-—–]?\s*', '', value, flags=re.I)
    return value.strip()


def strip_internal_asset_references(text: str, known_image_keys: Optional[List[str]] = None) -> str:
    value = text or ''
    normalized_known = [normalize_report_image_key(x) for x in (known_image_keys or []) if x]

    def _paren_repl(match: re.Match) -> str:
        label = re.sub(r'\s+', '', match.group(1))
        inner = match.group(2).strip()
        norm_inner = normalize_report_image_key(inner)
        if REPORT_IMAGE_ID_LIKE_RE.search(inner) or any(k and (k in norm_inner or norm_inner in k) for k in normalized_known):
            return label
        return match.group(0)

    value = re.sub(
        r'((?:图|表)\s*[0-9一二三四五六七八九十百千万]+)\s*[（(]\s*([^）)]{3,220})\s*[）)]',
        _paren_repl,
        value,
    )
    value = re.sub(
        r'\b(?:Figure|Fig\.?|Table)\s*([0-9]+)\s*[（(]\s*([^）)]{3,220})\s*[）)]',
        lambda m: f"{'表' if m.group(0).lower().startswith('table') else '图'}{m.group(1)}"
        if REPORT_IMAGE_ID_LIKE_RE.search(m.group(2)) else m.group(0),
        value,
        flags=re.I,
    )
    return value


def replace_report_label_aliases(text: str, aliases: Dict[str, str]) -> str:
    value = text or ''
    if not value or not aliases:
        return value

    sorted_items = [
        (old_label, new_label)
        for old_label, new_label in sorted(
            (aliases or {}).items(),
            key=lambda item: (-len(item[0]), item[0])
        )
        if old_label and new_label and old_label != new_label
    ]
    if not sorted_items:
        return value

    placeholder_to_new_label: Dict[str, str] = {}

    def _build_patterns(old_label: str) -> List[re.Pattern]:
        kind = old_label[0]
        old_number = old_label[1:].strip()
        variants = report_number_variants(old_number) or [old_number]
        patterns: List[re.Pattern] = []

        for variant in sorted(set(variants), key=len, reverse=True):
            escaped = re.escape(variant)
            if kind == '表':
                patterns.append(re.compile(rf'\b(?:Table)\s*{escaped}\b', flags=re.I))
                patterns.append(re.compile(
                    rf'表\s*{escaped}(?![0-9A-Za-z一二三四五六七八九十百千万零〇两])'
                ))
            else:
                patterns.append(re.compile(rf'\b(?:Figure|Fig\.?)\s*{escaped}\b', flags=re.I))
                patterns.append(re.compile(
                    rf'图\s*{escaped}(?![0-9A-Za-z一二三四五六七八九十百千万零〇两])'
                ))
        return patterns

    for idx, (old_label, new_label) in enumerate(sorted_items):
        placeholder = f"__REPORT_LABEL_SWAP_{idx}__"
        placeholder_to_new_label[placeholder] = new_label
        for pattern in _build_patterns(old_label):
            value = pattern.sub(placeholder, value)

    for placeholder, new_label in placeholder_to_new_label.items():
        value = value.replace(placeholder, new_label)

    return value


def collect_report_labels_from_text(text: str) -> List[str]:
    labels: List[str] = []
    for kind, number in REPORT_FIG_TABLE_LABEL_RE.findall(text or ''):
        labels.append(normalize_report_label(kind, number))
    for kind_text, number in REPORT_EN_LABEL_RE.findall(text or ''):
        kind = '表' if kind_text.lower().startswith('table') else '图'
        labels.append(normalize_report_label(kind, number))
    return labels


def is_standalone_figure_table_caption_line(line: str) -> bool:
    stripped = (line or '').strip()
    if not stripped:
        return False
    if REPORT_IMAGE_LINE_RE.match(stripped):
        return True
    if re.match(r'^(?:图|表)\s*[0-9一二三四五六七八九十百千万]+\s*[：:：.．\-—–]', stripped):
        return True
    if re.match(r'^(?:Figure|Fig\.?|Table)\s*[0-9]+\s*[：:：.．\-—–]', stripped, flags=re.I):
        return True
    return False


def build_report_asset_maps(
    report_md: str,
    image_ids: Optional[List[str]] = None,
    vision_summaries: str = '',
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    available_keys = list(image_ids or [])
    vision_meta = parse_vision_figure_metadata(vision_summaries)
    key_meta: Dict[str, Dict[str, str]] = {}

    for key in available_keys:
        norm_key = normalize_report_image_key(key)
        key_meta[norm_key] = {'key': key, 'kind': '图', 'number': '', 'caption': '', 'type_text': '', 'original_text': ''}
        if norm_key in vision_meta:
            key_meta[norm_key].update(vision_meta[norm_key])
            key_meta[norm_key]['key'] = key

    for line in normalize_report_markdown(report_md).splitlines():
        match = REPORT_IMAGE_LINE_RE.match(line.strip())
        if not match:
            continue
        key = match.group('key').strip()
        norm_key = normalize_report_image_key(key)
        if norm_key not in key_meta:
            key_meta[norm_key] = {'key': key, 'kind': '图', 'number': '', 'caption': '', 'type_text': '', 'original_text': ''}
        caption = strip_internal_asset_references(match.group('caption').strip(), available_keys)
        label = extract_report_label(caption)
        if label:
            if key_meta[norm_key].get('kind') != '表':
                key_meta[norm_key]['kind'] = label[0]
            key_meta[norm_key]['number'] = key_meta[norm_key].get('number') or label[1]
        if caption and not key_meta[norm_key].get('caption'):
            key_meta[norm_key]['caption'] = caption
        if is_table_like_figure_text(caption):
            key_meta[norm_key]['kind'] = '表'

    counters = {'图': 1, '表': 1}
    used_numbers = {'图': set(), '表': set()}
    for meta in key_meta.values():
        kind = meta.get('kind') or '图'
        number = str(meta.get('number') or '').strip()
        if number:
            used_numbers.setdefault(kind, set()).add(number)
            if number.isdigit():
                counters[kind] = max(counters.get(kind, 1), int(number) + 1)

    key_to_label: Dict[str, str] = {}
    label_to_key: Dict[str, str] = {}
    label_to_caption: Dict[str, str] = {}
    aliases: Dict[str, str] = {}

    for norm_key, meta in key_meta.items():
        kind = meta.get('kind') or '图'
        number = str(meta.get('number') or '').strip()
        if not number:
            while str(counters[kind]) in used_numbers.setdefault(kind, set()):
                counters[kind] += 1
            number = str(counters[kind])
            used_numbers[kind].add(number)
            counters[kind] += 1
        label = normalize_report_label(kind, number)
        key_to_label[norm_key] = label
        label_to_key.setdefault(label, meta.get('key') or norm_key)
        caption_core = strip_label_prefix(strip_internal_asset_references(meta.get('caption') or '', available_keys))
        label_to_caption[label] = caption_core

    for line in normalize_report_markdown(report_md).splitlines():
        match = REPORT_IMAGE_LINE_RE.match(line.strip())
        if not match:
            continue
        norm_key = normalize_report_image_key(match.group('key').strip())
        current_label = extract_report_label(match.group('caption').strip())
        new_label = key_to_label.get(norm_key)
        if current_label and new_label:
            old_label = normalize_report_label(*current_label)
            if old_label != new_label:
                aliases[old_label] = new_label

    return key_to_label, label_to_key, label_to_caption, aliases


def build_report_image_markdown(label: str, key: str, label_to_caption: Dict[str, str]) -> str:
    caption_core = strip_label_prefix(label_to_caption.get(label, '')).strip()
    caption = f"{label}：{caption_core}" if caption_core else label
    return f"![{caption}]({key})"

def build_report_order_maps_from_image_blocks(
    filtered_lines: List[str],
    key_to_label: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    按最终报告里图片块的真实出现顺序重新连续编号。
    关键要求：
    1. 先整体建立 old_label -> new_label 的映射
    2. 同一主图下的子图后缀必须保留，如 图4A -> 图2A、图4B -> 图2B
    3. 同一主图编号只在第一次出现时推进一次主计数
    返回：
    1. old_label -> new_label
    2. norm_key -> new_label
    """

    def _split_label(label: str) -> Tuple[str, str, str]:
        value = (label or '').strip()
        if not value:
            return '图', '', ''
        kind = value[0] if value[0] in {'图', '表'} else '图'
        body = value[1:].strip()

        m = re.fullmatch(
            r'([0-9IVXLCDM一二三四五六七八九十百千万零〇两]+)([A-Za-z]?)',
            body,
            flags=re.I,
        )
        if not m:
            return kind, body, ''
        main_number = m.group(1)
        suffix = (m.group(2) or '').upper()
        return kind, main_number, suffix

    label_aliases: Dict[str, str] = {}
    key_new_labels: Dict[str, str] = {}
    counters = {'图': 1, '表': 1}
    main_number_aliases: Dict[Tuple[str, str], str] = {}

    for line in filtered_lines:
        match = REPORT_IMAGE_LINE_RE.match(line.strip())
        if not match:
            continue

        key = match.group('key').strip()
        norm_key = normalize_report_image_key(key)

        old_label = key_to_label.get(norm_key)
        if not old_label:
            caption_label = extract_report_label(match.group('caption').strip())
            old_label = normalize_report_label(*caption_label) if caption_label else ''

        if not old_label:
            continue

        kind, old_main_number, suffix = _split_label(old_label)
        if kind not in counters:
            kind = '图'

        main_key = (kind, old_main_number)
        if main_key not in main_number_aliases:
            main_number_aliases[main_key] = str(counters[kind])
            counters[kind] += 1

        new_main_number = main_number_aliases[main_key]
        new_number = f"{new_main_number}{suffix}" if suffix else new_main_number
        new_label = normalize_report_label(kind, new_number)

        key_new_labels[norm_key] = new_label
        if old_label and old_label not in label_aliases:
            label_aliases[old_label] = new_label

    return label_aliases, key_new_labels

def reconcile_report_figure_table_references(
    report_md: str,
    image_ids: Optional[List[str]] = None,
    vision_summaries: str = '',
) -> str:
    text = normalize_report_markdown(report_md)
    if not text:
        return text

    available_keys = list(image_ids or [])
    key_to_label, label_to_key, label_to_caption, aliases = build_report_asset_maps(
        text,
        available_keys,
        vision_summaries,
    )
    known_keys = available_keys + list(label_to_key.values())

    cleaned_lines: List[str] = []
    cited_labels = set()
    for line in text.splitlines():
        stripped = line.strip()
        img_match = REPORT_IMAGE_LINE_RE.match(stripped)
        if img_match:
            cleaned_lines.append(line)
            continue

        cleaned = strip_internal_asset_references(line, known_keys)
        cleaned = replace_report_label_aliases(cleaned, aliases)
        cleaned_lines.append(cleaned)

        if not is_standalone_figure_table_caption_line(cleaned):
            cited_labels.update(collect_report_labels_from_text(cleaned))

    inserted_labels = set()
    used_image_keys = set()
    filtered_lines: List[str] = []
    for line in cleaned_lines:
        stripped = line.strip()
        img_match = REPORT_IMAGE_LINE_RE.match(stripped)
        if not img_match:
            filtered_lines.append(line)
            continue

        caption = strip_internal_asset_references(img_match.group('caption').strip(), known_keys)
        key = img_match.group('key').strip()
        norm_key = normalize_report_image_key(key)
        label = key_to_label.get(norm_key)
        if not label:
            caption_label = extract_report_label(caption)
            label = normalize_report_label(*caption_label) if caption_label else ''

        if label and label in cited_labels and norm_key not in used_image_keys:
            canonical_key = label_to_key.get(label, key)
            filtered_lines.append(build_report_image_markdown(label, canonical_key, label_to_caption))
            inserted_labels.add(label)
            used_image_keys.add(normalize_report_image_key(canonical_key))
        else:
            continue

    missing_labels = [
        label
        for label in sorted(
            cited_labels,
            key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 10**9, x)
        )
        if label in label_to_key and label not in inserted_labels
    ]

    for label in missing_labels:
        image_line = build_report_image_markdown(label, label_to_key[label], label_to_caption)
        inserted = False
        for idx, line in enumerate(filtered_lines):
            if REPORT_IMAGE_LINE_RE.match(line.strip()) or is_standalone_figure_table_caption_line(line):
                continue
            if label in collect_report_labels_from_text(line):
                filtered_lines.insert(idx + 1, '')
                filtered_lines.insert(idx + 2, image_line)
                inserted = True
                break
        if not inserted:
            filtered_lines.append('')
            filtered_lines.append(image_line)

    # 关键新增：
    # 按最终图片块出现顺序重新编号，并把正文里的“图4/表4”同步改成新的“图2/表2”。
    order_aliases, key_new_labels = build_report_order_maps_from_image_blocks(filtered_lines, key_to_label)

    final_lines: List[str] = []
    for line in filtered_lines:
        stripped = line.strip()
        img_match = REPORT_IMAGE_LINE_RE.match(stripped)

        if not img_match:
            final_lines.append(replace_report_label_aliases(line, order_aliases))
            continue

        key = img_match.group('key').strip()
        norm_key = normalize_report_image_key(key)
        old_label = key_to_label.get(norm_key)

        if not old_label:
            caption_label = extract_report_label(img_match.group('caption').strip())
            old_label = normalize_report_label(*caption_label) if caption_label else ''

        new_label = key_new_labels.get(norm_key, order_aliases.get(old_label, old_label))

        caption_core = strip_label_prefix(label_to_caption.get(old_label, '')).strip()
        if not caption_core:
            caption_core = strip_label_prefix(
                strip_internal_asset_references(img_match.group('caption').strip(), known_keys)
            ).strip()

        new_caption = f"{new_label}：{caption_core}" if (new_label and caption_core) else (new_label or img_match.group('caption').strip())
        final_lines.append(f"![{new_caption}]({key})")

    return normalize_report_markdown('\n'.join(final_lines))

def postprocess_generated_report_markdown(
    md_text: str,
    image_ids: Optional[List[str]] = None,
    vision_summaries: str = '',
) -> str:
    doc_title, body = split_title_and_body(md_text)
    blocks = split_markdown_blocks(body)
    cleaned_blocks: List[Tuple[str, object]] = []

    for idx, (block_type, payload) in enumerate(blocks):
        prev = cleaned_blocks[-1] if cleaned_blocks else None
        next_block = blocks[idx + 1] if idx + 1 < len(blocks) else None

        if block_type == 'table_title':
            current_title = normalize_table_title_line(str(payload))
            if prev and prev[0] == 'table_title' and captions_equivalent(current_title, str(prev[1])):
                continue
            cleaned_blocks.append(('table_title', current_title))
            continue

        if block_type == 'paragraph':
            paragraph_text = str(payload).strip()
            if prev and prev[0] == 'table_title' and captions_equivalent(paragraph_text, str(prev[1])):
                continue
            if is_table_title_line(paragraph_text) and next_block and next_block[0] in {'image', 'md_table'}:
                normalized_title = normalize_table_title_line(paragraph_text)
                if not (prev and prev[0] == 'table_title' and captions_equivalent(normalized_title, str(prev[1]))):
                    cleaned_blocks.append(('table_title', normalized_title))
                continue
            cleaned_blocks.append((block_type, paragraph_text))
            continue

        if block_type == 'image':
            caption, key = payload
            caption_text = (caption or '').strip()
            if prev and prev[0] == 'table_title' and caption_text and captions_equivalent(caption_text, str(prev[1])):
                cleaned_blocks.append(('image', ('', key)))
                continue
            cleaned_blocks.append(('image', (caption_text, key)))
            continue

        cleaned_blocks.append((block_type, payload))

    cleaned_blocks = deduplicate_report_image_blocks(cleaned_blocks)
    serialized = serialize_report_blocks(cleaned_blocks, doc_title)
    return reconcile_report_figure_table_references(serialized, image_ids=image_ids, vision_summaries=vision_summaries)


def captions_equivalent(left: str, right: str) -> bool:
    norm_left = normalize_caption_core_text(left) or normalize_compare_text(left)
    norm_right = normalize_caption_core_text(right) or normalize_compare_text(right)
    if not norm_left or not norm_right:
        return False
    return norm_left == norm_right or norm_left in norm_right or norm_right in norm_left


# ==========================================
# 模块 12：PDF 文档模板
# ==========================================


# ==========================================
# 模块 13：用户账户与历史报告持久化
# ==========================================
import uuid

try:
    import psycopg
    _PG_DRIVER = "psycopg"
except Exception:
    psycopg = None
    try:
        import psycopg2
        _PG_DRIVER = "psycopg2"
    except Exception:
        psycopg2 = None
        _PG_DRIVER = None

SUPABASE_DB_URL = (st.secrets.get("SUPABASE_DB_URL", "") or "").strip()
ASYNC_MODAL_API_URL = (st.secrets.get("ASYNC_MODAL_API_URL", "") or st.secrets.get("MODAL_JOB_API_URL", "") or "").strip()
PASSWORD_HASH_ROUNDS = 120000
JOB_STATUS_REFRESH_INTERVAL_MS = 180000
DB_READ_CACHE_TTL_SECONDS = 6


def normalize_supabase_db_url(db_url: str) -> str:
    value = (db_url or "").strip()
    if value and "sslmode=" not in value:
        value = f"{value}{'&' if '?' in value else '?'}sslmode=require"
    return value


SUPABASE_DB_URL = normalize_supabase_db_url(SUPABASE_DB_URL)


def _open_db_connection(db_url: str):
    if not db_url:
        raise RuntimeError("未在 Streamlit secrets 中配置 SUPABASE_DB_URL。")
    if _PG_DRIVER == "psycopg":
        return psycopg.connect(db_url)
    if _PG_DRIVER == "psycopg2":
        return psycopg2.connect(db_url)
    raise RuntimeError("未检测到 PostgreSQL 驱动。请在 requirements 中安装 psycopg[binary] 或 psycopg2-binary。")


def get_db_connection():
    return _open_db_connection(SUPABASE_DB_URL)


def _rows_to_dicts(cursor, rows):
    columns = [desc[0] for desc in (cursor.description or [])]
    return [dict(zip(columns, row)) for row in rows]


def _execute_bootstrap_statements(db_url: str):
    statements = [
        "CREATE EXTENSION IF NOT EXISTS pgcrypto",
        """
        CREATE TABLE IF NOT EXISTS public.users (
            id UUID PRIMARY KEY,
            username TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_lower ON public.users ((LOWER(username)))",
        """
        CREATE TABLE IF NOT EXISTS public.analysis_jobs (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
            task_no INTEGER NOT NULL,
            title TEXT,
            pdf_name TEXT,
            pdf_path TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            progress_text TEXT DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "ALTER TABLE public.analysis_jobs ADD COLUMN IF NOT EXISTS cache_key TEXT",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_analysis_jobs_user_task_unique ON public.analysis_jobs(user_id, task_no)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_analysis_jobs_user_cache_key ON public.analysis_jobs(user_id, cache_key) WHERE cache_key IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS idx_analysis_jobs_user_id ON public.analysis_jobs(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON public.analysis_jobs(status)",
        """
        CREATE TABLE IF NOT EXISTS public.analysis_reports (
            id UUID PRIMARY KEY,
            job_id UUID NOT NULL UNIQUE REFERENCES public.analysis_jobs(id) ON DELETE CASCADE,
            user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
            report_markdown TEXT,
            parsed_markdown TEXT,
            text_agent_output TEXT,
            vision_output TEXT,
            images_manifest JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_analysis_reports_user_id ON public.analysis_reports(user_id)",
        """
        CREATE TABLE IF NOT EXISTS public.analysis_agent_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id UUID NOT NULL REFERENCES public.analysis_jobs(id) ON DELETE CASCADE,
            step_no INTEGER NOT NULL,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT,
            instructions TEXT,
            expected_output TEXT,
            status TEXT NOT NULL DEFAULT 'finished',
            details JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_analysis_agent_logs_job_step ON public.analysis_agent_logs(job_id, step_no)",
        """
        CREATE TABLE IF NOT EXISTS public.paper_search_jobs (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
            topic TEXT NOT NULL DEFAULT '',
            requirements TEXT NOT NULL DEFAULT '',
            preprint_rule TEXT NOT NULL DEFAULT '',
            feedback TEXT NOT NULL DEFAULT '',
            previous_result TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'queued',
            progress_text TEXT NOT NULL DEFAULT '',
            result_markdown TEXT NOT NULL DEFAULT '',
            agent_logs JSONB NOT NULL DEFAULT '[]'::jsonb,
            raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            is_final BOOLEAN NOT NULL DEFAULT FALSE,
            finalized_at TIMESTAMPTZ,
            superseded_by UUID,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_paper_search_jobs_user_created ON public.paper_search_jobs(user_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_paper_search_jobs_status ON public.paper_search_jobs(status)",
        "ALTER TABLE public.paper_search_jobs ADD COLUMN IF NOT EXISTS is_final BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE public.paper_search_jobs ADD COLUMN IF NOT EXISTS finalized_at TIMESTAMPTZ",
        "ALTER TABLE public.paper_search_jobs ADD COLUMN IF NOT EXISTS superseded_by UUID",
        "CREATE INDEX IF NOT EXISTS idx_paper_search_jobs_user_final ON public.paper_search_jobs(user_id, is_final, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_paper_search_jobs_superseded ON public.paper_search_jobs(superseded_by)",
        """
        CREATE TABLE IF NOT EXISTS public.paper_search_selected_papers (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            search_job_id UUID NOT NULL REFERENCES public.paper_search_jobs(id) ON DELETE CASCADE,
            user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
            rank_no INTEGER NOT NULL CHECK (rank_no BETWEEN 1 AND 6),
            title TEXT NOT NULL,
            venue TEXT NOT NULL DEFAULT '',
            doi TEXT NOT NULL DEFAULT '',
            s2_id TEXT NOT NULL DEFAULT '',
            abstract TEXT NOT NULL DEFAULT '',
            recommendation_reason TEXT NOT NULL DEFAULT '',
            dedupe_key TEXT NOT NULL,
            raw_item JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_search_selected_job_rank ON public.paper_search_selected_papers(search_job_id, rank_no)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_search_selected_user_dedupe ON public.paper_search_selected_papers(user_id, dedupe_key)",
        "CREATE INDEX IF NOT EXISTS idx_paper_search_selected_user_created ON public.paper_search_selected_papers(user_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_paper_search_selected_user_doi ON public.paper_search_selected_papers(user_id, doi) WHERE doi <> ''",
        "CREATE INDEX IF NOT EXISTS idx_paper_search_selected_user_s2 ON public.paper_search_selected_papers(user_id, s2_id) WHERE s2_id <> ''",
    ]

    conn = _open_db_connection(db_url)
    try:
        with conn.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return True


@st.cache_resource(show_spinner=False)
def _bootstrap_database_once(db_url: str):
    return _execute_bootstrap_statements(db_url)


def ensure_app_storage():
    if not SUPABASE_DB_URL:
        raise RuntimeError("未在 Streamlit secrets 中配置 SUPABASE_DB_URL。")
    _bootstrap_database_once(SUPABASE_DB_URL)


def db_fetch_one(query: str, params: Optional[Tuple[Any, ...]] = None) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params or ())
            row = cursor.fetchone()
            if row is None:
                return None
            return _rows_to_dicts(cursor, [row])[0]
    finally:
        conn.close()


def db_fetch_all(query: str, params: Optional[Tuple[Any, ...]] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params or ())
            rows = cursor.fetchall()
            return _rows_to_dicts(cursor, rows)
    finally:
        conn.close()


def format_db_timestamp(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime.datetime):
        if value.tzinfo:
            value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return ""
        try:
            parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo:
                parsed = parsed.astimezone(datetime.timezone.utc).replace(tzinfo=None)
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return value[:19].replace("T", " ")
    return str(value)


def normalize_json_field(value: Any, default: Any):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


def clear_db_read_caches():
    _get_user_record_cached.clear()
    _load_user_report_index_cached.clear()
    _get_user_job_state_cached.clear()
    _get_user_job_by_cache_key_cached.clear()
    _load_user_report_record_cached.clear()
    _get_user_cached_report_cached.clear()
    for cache_func_name in [
        '_load_user_search_index_cached',
        '_get_user_search_job_state_cached',
        '_load_user_search_record_cached',
    ]:
        cache_func = globals().get(cache_func_name)
        if cache_func is not None and hasattr(cache_func, 'clear'):
            cache_func.clear()


def db_execute(query: str, params: Optional[Tuple[Any, ...]] = None):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params or ())
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    clear_db_read_caches()


def normalize_username(username: str) -> str:
    return (username or "").strip()


def canonical_username(username: str) -> str:
    return normalize_username(username).lower()


def make_password_hash(password: str, salt_hex: Optional[str] = None) -> Tuple[str, str]:
    salt_hex = salt_hex or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        (password or "").encode("utf-8"),
        bytes.fromhex(salt_hex),
        PASSWORD_HASH_ROUNDS,
    ).hex()
    return salt_hex, digest


def pack_password_hash(password: str) -> str:
    salt_hex, digest = make_password_hash(password)
    return f"{salt_hex}${digest}"


def verify_password_hash(password: str, packed_hash: str) -> bool:
    packed_hash = (packed_hash or "").strip()
    if "$" not in packed_hash:
        return False
    salt_hex, expected_hash = packed_hash.split("$", 1)
    _, actual_hash = make_password_hash(password, salt_hex=salt_hex)
    return actual_hash == expected_hash


@st.cache_data(show_spinner=False, ttl=DB_READ_CACHE_TTL_SECONDS)
def _get_user_record_cached(username: str) -> Optional[Dict[str, Any]]:
    return db_fetch_one(
        """
        SELECT id, username, password_hash, created_at
        FROM public.users
        WHERE LOWER(username) = LOWER(%s)
        LIMIT 1
        """,
        (username,),
    )


def get_user_record(username: str) -> Optional[Dict[str, Any]]:
    username = normalize_username(username)
    if not username:
        return None
    ensure_app_storage()
    return _get_user_record_cached(username)


def register_user(username: str, password: str) -> Tuple[bool, str]:
    username = normalize_username(username)
    password = password or ""

    if not username:
        return False, "请输入账号。"
    if len(username) < 3:
        return False, "账号至少需要 3 个字符。"
    if len(username) > 32:
        return False, "账号长度请控制在 32 个字符以内。"
    if len(password) < 6:
        return False, "密码至少需要 6 个字符。"

    try:
        ensure_app_storage()
        if get_user_record(username):
            return False, "该账号已存在，请直接登录。"

        db_execute(
            """
            INSERT INTO public.users (id, username, password_hash, created_at)
            VALUES (%s, %s, %s, NOW())
            """,
            (str(uuid.uuid4()), username, pack_password_hash(password)),
        )
        return True, username
    except Exception as e:
        if "idx_users_username_lower" in str(e) or "duplicate key" in str(e).lower():
            return False, "该账号已存在，请直接登录。"
        return False, f"注册失败：{str(e)}"


def authenticate_user(username: str, password: str) -> Tuple[bool, str]:
    username = normalize_username(username)
    password = password or ""

    try:
        ensure_app_storage()
        record = get_user_record(username)
    except Exception as e:
        return False, f"登录失败：{str(e)}"

    if not record:
        return False, "账号或密码错误。"
    if not verify_password_hash(password, record.get("password_hash", "")):
        return False, "账号或密码错误。"
    return True, record.get("username") or username


def get_history_selector_key(username: str) -> str:
    return f"history_selector_{canonical_username(username)}"


def get_history_reset_flag_key(username: str) -> str:
    return f"history_reset_pending_{canonical_username(username)}"


def reset_user_workspace_view(username: Optional[str] = None):
    username = normalize_username(username or st.session_state.get("current_user", ""))
    if not username:
        return
    reset_flag_key = get_history_reset_flag_key(username)
    st.session_state[reset_flag_key] = True
    st.session_state.selected_history_report_id = None
    st.session_state.selected_history_search_id = None


def start_fresh_workspace(username: Optional[str] = None):
    username = normalize_username(username or st.session_state.get("current_user", ""))
    reset_user_workspace_view(username)
    st.session_state.app_state = "IDLE"
    st.session_state.final_result = ""
    st.session_state.has_provided_feedback = False
    st.session_state.ui_logs = []
    st.session_state.feedback_start_time = None
    st.session_state.active_search_job_id = ""
    st.session_state.search_topic = ""
    st.session_state.search_requirements = ""
    st.session_state.search_preprint_rule = "排除预印本 (仅限正规期刊/会议)"
    st.session_state.sidebar_direct_entries = []
    st.session_state.bottom_direct_entries = []


def get_user_space_dir(username: str) -> str:
    return canonical_username(username)


def build_report_meta_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "report_id": row.get("report_id") or row.get("job_id") or "",
        "cache_key": row.get("cache_key", "") or "",
        "source_name": row.get("source_name") or row.get("pdf_name") or row.get("report_title") or "未命名论文",
        "report_title": row.get("report_title") or row.get("title") or row.get("source_name") or "论文全维度深度透视报告",
        "created_at": format_db_timestamp(row.get("created_at") or row.get("job_created_at") or row.get("report_created_at")),
        "updated_at": format_db_timestamp(row.get("updated_at") or row.get("report_created_at") or row.get("job_created_at") or row.get("created_at")),
        "status": (row.get("status") or "").lower(),
        "progress_text": row.get("progress_text") or "",
        "has_report": bool(row.get("has_report")),
        "task_no": row.get("task_no"),
    }


@st.cache_data(show_spinner=False, ttl=DB_READ_CACHE_TTL_SECONDS)
def _load_user_report_index_cached(username: str) -> List[Dict[str, Any]]:
    rows = db_fetch_all(
        """
        SELECT
            j.id AS report_id,
            j.task_no,
            j.cache_key,
            COALESCE(NULLIF(j.pdf_name, ''), NULLIF(j.title, ''), '未命名论文') AS source_name,
            COALESCE(NULLIF(j.title, ''), NULLIF(j.pdf_name, ''), '论文全维度深度透视报告') AS report_title,
            j.status,
            j.progress_text,
            j.created_at,
            COALESCE(j.updated_at, r.created_at, j.created_at) AS updated_at,
            CASE WHEN r.job_id IS NOT NULL THEN TRUE ELSE FALSE END AS has_report
        FROM public.analysis_jobs j
        JOIN public.users u ON u.id = j.user_id
        LEFT JOIN public.analysis_reports r ON r.job_id = j.id
        WHERE LOWER(u.username) = LOWER(%s)
        ORDER BY COALESCE(j.updated_at, r.created_at, j.created_at) DESC
        """,
        (username,),
    )
    return [build_report_meta_from_row(row) for row in rows]


def load_user_report_index(username: str) -> List[Dict[str, Any]]:
    username = normalize_username(username)
    if not username:
        return []
    ensure_app_storage()
    return _load_user_report_index_cached(username)


def get_persistable_analysis_result(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_markdown": analysis_result.get("source_markdown", ""),
        "text_report": analysis_result.get("text_report", ""),
        "vision_summaries": analysis_result.get("vision_summaries", ""),
        "images": analysis_result.get("images", {}),
        "main_report": analysis_result.get("main_report", ""),
    }


def get_next_task_no(user_id: str) -> int:
    row = db_fetch_one(
        "SELECT COALESCE(MAX(task_no), 0) + 1 AS next_task_no FROM public.analysis_jobs WHERE user_id = %s",
        (user_id,),
    )
    return int((row or {}).get("next_task_no") or 1)


@st.cache_data(show_spinner=False, ttl=DB_READ_CACHE_TTL_SECONDS)
def _get_user_job_state_cached(username: str, report_id: str) -> Optional[Dict[str, Any]]:
    row = db_fetch_one(
        """
        SELECT
            j.id AS report_id,
            j.task_no,
            j.cache_key,
            j.title AS report_title,
            j.pdf_name AS source_name,
            j.status,
            j.progress_text,
            j.created_at,
            j.updated_at,
            CASE WHEN r.job_id IS NOT NULL THEN TRUE ELSE FALSE END AS has_report
        FROM public.analysis_jobs j
        JOIN public.users u ON u.id = j.user_id
        LEFT JOIN public.analysis_reports r ON r.job_id = j.id
        WHERE LOWER(u.username) = LOWER(%s) AND j.id = %s
        LIMIT 1
        """,
        (username, report_id),
    )
    if not row:
        return None
    return build_report_meta_from_row(row)


def get_user_job_state(username: str, report_id: str) -> Optional[Dict[str, Any]]:
    username = normalize_username(username)
    report_id = (report_id or "").strip()
    if not username or not report_id:
        return None
    ensure_app_storage()
    return _get_user_job_state_cached(username, report_id)


@st.cache_data(show_spinner=False, ttl=DB_READ_CACHE_TTL_SECONDS)
def _get_user_job_by_cache_key_cached(username: str, cache_key: str) -> Optional[Dict[str, Any]]:
    row = db_fetch_one(
        """
        SELECT
            j.id AS report_id,
            j.task_no,
            j.cache_key,
            j.title AS report_title,
            j.pdf_name AS source_name,
            j.status,
            j.progress_text,
            j.created_at,
            j.updated_at,
            CASE WHEN r.job_id IS NOT NULL THEN TRUE ELSE FALSE END AS has_report
        FROM public.analysis_jobs j
        JOIN public.users u ON u.id = j.user_id
        LEFT JOIN public.analysis_reports r ON r.job_id = j.id
        WHERE LOWER(u.username) = LOWER(%s) AND j.cache_key = %s
        LIMIT 1
        """,
        (username, cache_key),
    )
    if not row:
        return None
    return build_report_meta_from_row(row)


def get_user_job_by_cache_key(username: str, cache_key: str) -> Optional[Dict[str, Any]]:
    username = normalize_username(username)
    cache_key = (cache_key or "").strip()
    if not username or not cache_key:
        return None
    ensure_app_storage()
    return _get_user_job_by_cache_key_cached(username, cache_key)


def focus_latest_user_job(username: str):
    username = normalize_username(username)
    selector_key = get_history_selector_key(username)
    reset_flag_key = get_history_reset_flag_key(username)
    history = load_user_report_index(username)
    if history:
        latest_id = history[0].get("report_id", "")
        if latest_id:
            st.session_state[selector_key] = latest_id
            st.session_state.selected_history_report_id = latest_id
            st.session_state[reset_flag_key] = False
            return
    reset_user_workspace_view(username)


def update_analysis_job_status(job_id: str, status: str, progress_text: str):
    job_id = (job_id or "").strip()
    if not job_id:
        return
    safe_status = (status or "").strip().lower()
    if safe_status not in {"queued", "processing", "finished", "failed"}:
        safe_status = "processing"
    db_execute(
        """
        UPDATE public.analysis_jobs
        SET status = %s,
            progress_text = %s,
            updated_at = NOW()
        WHERE id = %s
        """,
        (safe_status, (progress_text or "").strip()[:1000], job_id),
    )


def create_or_reuse_analysis_job(username: str, source_name: str, cache_key: str) -> Tuple[Dict[str, Any], bool]:
    ensure_app_storage()
    username = normalize_username(username)
    user_record = get_user_record(username)
    if not user_record:
        raise RuntimeError("当前账号不存在，无法创建后台解析任务。")

    source_name = (source_name or "").strip() or "未命名论文"
    user_id = user_record["id"]
    existing_job = get_user_job_by_cache_key(username, cache_key)
    if existing_job:
        existing_status = (existing_job.get("status") or "").lower()
        if existing_status in {"queued", "processing"}:
            return existing_job, False
        if existing_status == "finished" and existing_job.get("has_report"):
            return existing_job, False

        update_analysis_job_status(existing_job["report_id"], "queued", "任务已重新提交，等待后台启动")
        db_execute(
            """
            UPDATE public.analysis_jobs
            SET title = %s,
                pdf_name = %s,
                updated_at = NOW()
            WHERE id = %s
            """,
            (source_name, source_name, existing_job["report_id"]),
        )
        refreshed_job = get_user_job_state(username, existing_job["report_id"]) or existing_job
        return refreshed_job, True

    task_no = get_next_task_no(user_id)
    job_id = str(uuid.uuid4())
    db_execute(
        """
        INSERT INTO public.analysis_jobs (
            id, user_id, task_no, title, pdf_name, pdf_path, cache_key,
            status, progress_text, created_at, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'queued', '任务已提交，等待后台启动', NOW(), NOW())
        """,
        (job_id, user_id, task_no, source_name, source_name, "", cache_key),
    )
    created_job = get_user_job_state(username, job_id)
    if not created_job:
        raise RuntimeError("后台解析任务创建成功，但未能回读任务记录。")
    return created_job, True


def submit_analysis_job_to_modal(job_id: str, source_name: str, cache_key: str, pdf_bytes: bytes):
    submit_url = (ASYNC_MODAL_API_URL or "").strip()
    if not submit_url:
        raise RuntimeError("未在 Streamlit secrets 中配置 ASYNC_MODAL_API_URL。")

    files = {
        "file": (source_name or "paper.pdf", pdf_bytes, "application/pdf"),
    }
    data = {
        "task_type": "auto",
        "job_id": job_id,
        "source_name": source_name or "未命名论文",
        "cache_key": cache_key,
    }

    response = requests.post(submit_url, data=data, files=files, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"后台任务提交失败：HTTP {response.status_code}")

    try:
        payload = response.json()
    except Exception as e:
        raise RuntimeError(f"后台任务返回了无法解析的响应：{str(e)}")

    if payload.get("status") not in {"accepted", "queued", "processing"}:
        raise RuntimeError(payload.get("message") or "后台任务未被接受。")


def build_search_meta_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "search_job_id": str(row.get("search_job_id") or row.get("id") or ""),
        "topic": row.get("topic") or "论文检索",
        "requirements": row.get("requirements") or "",
        "preprint_rule": row.get("preprint_rule") or "",
        "feedback": row.get("feedback") or "",
        "status": row.get("status") or "queued",
        "progress_text": row.get("progress_text") or "",
        "is_final": bool(row.get("is_final")),
        "finalized_at": format_db_timestamp(row.get("finalized_at")),
        "superseded_by": str(row.get("superseded_by") or ""),
        "created_at": format_db_timestamp(row.get("created_at")),
        "updated_at": format_db_timestamp(row.get("updated_at")),
        "has_result": bool(row.get("result_markdown")),
    }


@st.cache_data(show_spinner=False, ttl=DB_READ_CACHE_TTL_SECONDS)
def _load_user_search_index_cached(username: str) -> List[Dict[str, Any]]:
    rows = db_fetch_all(
        """
        SELECT
            s.id AS search_job_id,
            s.topic,
            s.requirements,
            s.preprint_rule,
            s.feedback,
            s.status,
            s.progress_text,
            s.result_markdown,
            s.is_final,
            s.finalized_at,
            s.superseded_by,
            s.created_at,
            s.updated_at
        FROM public.paper_search_jobs s
        JOIN public.users u ON u.id = s.user_id
        WHERE LOWER(u.username) = LOWER(%s)
          AND s.superseded_by IS NULL
          AND (s.is_final = TRUE OR s.status IN ('queued', 'processing', 'finished', 'failed'))
        ORDER BY COALESCE(s.finalized_at, s.updated_at, s.created_at) DESC
        """,
        (username,),
    )
    return [build_search_meta_from_row(row) for row in rows]


def load_user_search_index(username: str) -> List[Dict[str, Any]]:
    username = normalize_username(username)
    if not username:
        return []
    ensure_app_storage()
    return _load_user_search_index_cached(username)


@st.cache_data(show_spinner=False, ttl=DB_READ_CACHE_TTL_SECONDS)
def _get_user_search_job_state_cached(username: str, search_job_id: str) -> Optional[Dict[str, Any]]:
    row = db_fetch_one(
        """
        SELECT
            s.id AS search_job_id,
            s.topic,
            s.requirements,
            s.preprint_rule,
            s.feedback,
            s.status,
            s.progress_text,
            s.result_markdown,
            s.is_final,
            s.finalized_at,
            s.superseded_by,
            s.created_at,
            s.updated_at
        FROM public.paper_search_jobs s
        JOIN public.users u ON u.id = s.user_id
        WHERE LOWER(u.username) = LOWER(%s) AND s.id = %s
        LIMIT 1
        """,
        (username, search_job_id),
    )
    if not row:
        return None
    return build_search_meta_from_row(row)


def get_user_search_job_state(username: str, search_job_id: str) -> Optional[Dict[str, Any]]:
    username = normalize_username(username)
    search_job_id = (search_job_id or "").strip()
    if not username or not search_job_id:
        return None
    ensure_app_storage()
    return _get_user_search_job_state_cached(username, search_job_id)


@st.cache_data(show_spinner=False, ttl=DB_READ_CACHE_TTL_SECONDS)
def _load_user_search_record_cached(username: str, search_job_id: str) -> Optional[Dict[str, Any]]:
    row = db_fetch_one(
        """
        SELECT
            s.id AS search_job_id,
            s.topic,
            s.requirements,
            s.preprint_rule,
            s.feedback,
            s.previous_result,
            s.status,
            s.progress_text,
            s.result_markdown,
            s.agent_logs,
            s.raw_payload,
            s.is_final,
            s.finalized_at,
            s.superseded_by,
            s.created_at,
            s.updated_at
        FROM public.paper_search_jobs s
        JOIN public.users u ON u.id = s.user_id
        WHERE LOWER(u.username) = LOWER(%s) AND s.id = %s
        LIMIT 1
        """,
        (username, search_job_id),
    )
    if not row:
        return None
    meta = build_search_meta_from_row(row)
    return {
        "meta": meta,
        "result_markdown": row.get("result_markdown") or "",
        "agent_logs": normalize_json_field(row.get("agent_logs"), []),
        "raw_payload": normalize_json_field(row.get("raw_payload"), {}),
    }


def load_user_search_record(username: str, search_job_id: str) -> Optional[Dict[str, Any]]:
    username = normalize_username(username)
    search_job_id = (search_job_id or "").strip()
    if not username or not search_job_id:
        return None
    ensure_app_storage()
    return _load_user_search_record_cached(username, search_job_id)


def create_paper_search_job(
    username: str,
    user_topic: str,
    user_requirements: str,
    preprint_rule: str,
    feedback: str = "",
    previous_result: str = "",
) -> Dict[str, Any]:
    ensure_app_storage()
    username = normalize_username(username)
    user_record = get_user_record(username)
    if not user_record:
        raise RuntimeError("当前账号不存在，无法创建论文检索任务。")

    job_id = str(uuid.uuid4())
    db_execute(
        """
        INSERT INTO public.paper_search_jobs (
            id, user_id, topic, requirements, preprint_rule,
            feedback, previous_result, status, progress_text,
            created_at, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'queued', '后台检索任务已创建，等待调度执行', NOW(), NOW())
        """,
        (
            job_id,
            user_record["id"],
            user_topic or "",
            user_requirements or "",
            preprint_rule or "排除预印本 (仅限正规期刊/会议)",
            feedback or "",
            previous_result or "",
        ),
    )
    created_job = get_user_search_job_state(username, job_id)
    if not created_job:
        raise RuntimeError("论文搜索任务创建成功，但未能回读任务记录。")
    return created_job


def update_paper_search_job_status(search_job_id: str, status: str, progress_text: str):
    search_job_id = (search_job_id or "").strip()
    if not search_job_id:
        return
    safe_status = (status or "").strip().lower()
    if safe_status not in {"queued", "processing", "finished", "failed"}:
        safe_status = "processing"
    db_execute(
        """
        UPDATE public.paper_search_jobs
        SET status = %s,
            progress_text = %s,
            updated_at = NOW()
        WHERE id = %s
        """,
        (safe_status, (progress_text or "").strip()[:1000], search_job_id),
    )


def submit_paper_search_to_modal(search_job_id: str):
    """把已创建的论文检索任务提交给后端统一 Director，由后端离线执行 Paper Search Agent。"""
    submit_url = (ASYNC_MODAL_API_URL or "").strip()
    if not submit_url:
        raise RuntimeError("未在 Streamlit secrets 中配置 ASYNC_MODAL_API_URL。")

    data = {
        "task_type": "paper_search",
        "job_id": search_job_id or "",
    }

    response = requests.post(submit_url, data=data, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"后台文献检索提交失败：HTTP {response.status_code}")

    try:
        payload = response.json()
    except Exception as e:
        raise RuntimeError(f"后端论文检索返回了无法解析的响应：{str(e)}")

    if payload.get("status") not in {"accepted", "queued", "processing"}:
        raise RuntimeError(payload.get("message") or payload.get("error") or "后端论文检索任务未被接受。")

    return payload


def finalize_paper_search_via_modal(search_job_id: str):
    """用户最终满意后，通知后端把当前 job 的六篇论文写入最终去重库。"""
    submit_url = (ASYNC_MODAL_API_URL or "").strip()
    if not submit_url:
        raise RuntimeError("未在 Streamlit secrets 中配置 ASYNC_MODAL_API_URL。")
    data = {"task_type": "paper_search_finalize", "job_id": search_job_id or ""}
    response = requests.post(submit_url, data=data, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"后端最终确认失败：HTTP {response.status_code}")
    try:
        payload = response.json()
    except Exception as e:
        raise RuntimeError(f"后端最终确认返回了无法解析的响应：{str(e)}")
    if payload.get("status") != "finished":
        raise RuntimeError(payload.get("message") or payload.get("error") or "后端未能最终确认论文搜索结果。")
    try:
        st.cache_data.clear()
    except Exception:
        pass
    return payload


def mark_paper_search_job_superseded(search_job_id: str, superseded_by: str):
    """用户不满意并创建新一轮反馈检索后，把上一轮标记为已被替代，避免作为最终历史记录展示。"""
    search_job_id = (search_job_id or "").strip()
    superseded_by = (superseded_by or "").strip()
    if not search_job_id or not superseded_by or search_job_id == superseded_by:
        return
    db_execute(
        """
        UPDATE public.paper_search_jobs
        SET superseded_by = %s,
            is_final = FALSE,
            progress_text = '该轮结果已由后续修正任务替代，未写入最终论文去重库',
            updated_at = NOW()
        WHERE id = %s AND is_final = FALSE
        """,
        (superseded_by, search_job_id),
    )
    db_execute("DELETE FROM public.paper_search_selected_papers WHERE search_job_id = %s", (search_job_id,))
    try:
        st.cache_data.clear()
    except Exception:
        pass



def build_search_ui_logs(search_logs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """把后端 Paper Search Agent 执行日志转换为前端展示用结构。"""
    ui_logs: List[Dict[str, str]] = []
    for idx, item in enumerate(search_logs or [], start=1):
        step = item.get("step") or idx
        thought_action = item.get("thought_action", "") or item.get("action", "") or ""
        observation = item.get("observation", "") or ""
        content_parts = []
        if thought_action:
            content_parts.append(f"**检索决策：**\n```text\n{thought_action}\n```")
        if observation:
            content_parts.append(f"**检索工具返回：**\n```text\n{observation}\n```")
        ui_logs.append({
            "title": f"文献检索 Agent 执行记录（第 {step} 步）",
            "content": "\n\n".join(content_parts) or "无详细日志。",
        })
    return ui_logs

def format_search_history_label(meta: Dict[str, Any]) -> str:
    if not meta:
        return "不打开检索记录"
    topic = shorten_sidebar_label(meta.get("topic") or "论文检索", max_len=18)
    status = (meta.get("status") or "").lower()
    if status in {"queued", "processing"}:
        return f"{topic}｜检索中"
    if status == "failed":
        return f"{topic}｜检索失败"
    if status == "finished" and not meta.get("is_final"):
        return f"{topic}｜待最终确认"
    timestamp = (meta.get("finalized_at") or meta.get("updated_at") or meta.get("created_at") or "")[:16]
    return f"{topic}｜{timestamp}" if timestamp else topic


def render_pending_search_notice(search_meta: Dict[str, Any]):
    topic = search_meta.get("topic") or "论文检索"
    status = (search_meta.get("status") or "").lower()
    progress_text = search_meta.get("progress_text") or "后台文献检索任务正在运行。"
    if status == "failed":
        st.error(progress_text or f"《{topic}》检索失败。")
        return
    st.info(f"《{topic}》当前状态：{progress_text}")
    st.caption("该检索任务已由后台接管。页面每 3 分钟自动刷新一次；关闭页面后仍可稍后重新登录查看结果。")



def render_saved_search_record(username: str, search_job_id: str):
    record = load_user_search_record(username, search_job_id)
    if not record:
        st.error("未找到该文献检索档案，可能已被删除。")
        return
    meta = record.get("meta", {}) or {}
    status = (meta.get("status") or "").lower()
    if status in {"queued", "processing"}:
        render_pending_search_notice(meta)
        st_autorefresh(interval=JOB_STATUS_REFRESH_INTERVAL_MS, key=f"search_history_refresh_{search_job_id}")
        return
    if status == "failed":
        st.error(meta.get("progress_text") or "该论文检索任务失败。")
        return
    if status == "finished" and not meta.get("is_final"):
        st.session_state.final_result = record.get("result_markdown", "") or "该检索任务暂无结果。"
        st.session_state.ui_logs = build_search_ui_logs(record.get("agent_logs", []) or [])
        st.session_state.current_search_job_id = search_job_id
        st.session_state.search_topic = meta.get("topic") or ""
        st.session_state.search_requirements = meta.get("requirements") or ""
        st.session_state.search_preprint_rule = meta.get("preprint_rule") or "排除预印本 (仅限正规期刊/会议)"
        st.session_state.feedback_start_time = time.time()
        st.session_state.app_state = "WAITING_FEEDBACK"
        st.rerun()

    st.markdown(f"### 文献检索档案：{meta.get('topic') or '论文检索'}")
    if meta.get("requirements"):
        with st.expander("查看筛选约束", expanded=False):
            st.markdown(meta.get("requirements") or "")
    logs = build_search_ui_logs(record.get("agent_logs", []) or [])
    if logs:
        st.markdown("#### 检索 Agent 执行轨迹")
        for log in logs:
            with st.expander(log["title"], expanded=False):
                st.markdown(log["content"])
    st.markdown("#### 六篇候选文献")
    with st.container(border=True):
        st.markdown(record.get("result_markdown") or "该历史检索暂无结果。")


def poll_active_paper_search_job(username: str, search_job_id: str) -> Optional[Dict[str, Any]]:
    """轮询当前后台论文搜索任务。"""
    meta = get_user_search_job_state(username, search_job_id)
    if not meta:
        return None
    status = (meta.get("status") or "").lower()
    if status == "finished":
        record = load_user_search_record(username, search_job_id)
        if record:
            st.session_state.final_result = record.get("result_markdown", "") or "后端未返回有效检索结果。"
            st.session_state.ui_logs = build_search_ui_logs(record.get("agent_logs", []) or [])
            st.session_state.app_state = "WAITING_FEEDBACK"
            st.session_state.feedback_start_time = time.time()
            st.session_state.current_search_job_id = search_job_id
            st.session_state.active_search_job_id = ""
            return record
    elif status == "failed":
        st.session_state.app_state = "IDLE"
        st.session_state.active_search_job_id = ""
    return {"meta": meta}


def _load_agent_logs_cached(username: str, report_id: str) -> List[Dict[str, Any]]:
    """从数据库读取指定任务的多 Agent 动作轨迹。"""
    try:
        return db_fetch_all(
            """
            SELECT
                l.step_no,
                l.actor,
                l.action,
                l.reason,
                l.instructions,
                l.expected_output,
                l.status,
                l.details,
                l.created_at
            FROM public.analysis_agent_logs l
            JOIN public.analysis_jobs j ON j.id = l.job_id
            JOIN public.users u ON u.id = j.user_id
            WHERE LOWER(u.username) = LOWER(%s)
              AND j.id = %s
            ORDER BY l.step_no ASC, l.created_at ASC
            """,
            (username, report_id),
        )
    except Exception:
        return []


def load_agent_logs(username: str, report_id: str) -> List[Dict[str, Any]]:
    """读取多 Agent 动作轨迹；用户名或任务为空时返回空列表。"""
    username = normalize_username(username)
    report_id = (report_id or "").strip()
    if not username or not report_id:
        return []
    ensure_app_storage()
    return _load_agent_logs_cached(username, report_id)


def format_agent_log_details(details: Any) -> str:
    """把日志 details 字段格式化为可读 JSON 文本。"""
    data = normalize_json_field(details, {}) if not isinstance(details, dict) else details
    if not data:
        return ""
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def render_agent_action_logs(username: str, report_id: str, expanded: bool = False):
    """在前端展示完整的多 Agent / 工具动作轨迹。"""
    logs = load_agent_logs(username, report_id)
    if not logs:
        st.caption("暂无完整 Agent 动作轨迹。新任务启动后会逐步写入这里。")
        return

    st.markdown("#### 多 Agent 动作轨迹")
    with st.expander("查看完整动作轨迹", expanded=expanded):
        for row in logs:
            step_no = row.get("step_no") or ""
            actor = row.get("actor") or "未知执行者"
            action = row.get("action") or "未知动作"
            status = row.get("status") or ""
            created_at = format_db_timestamp(row.get("created_at"))
            st.markdown(f"**{step_no}. {actor}｜{action}｜{status}**  {created_at}")
            if row.get("reason"):
                st.markdown(f"- **原因：** {row.get('reason')}")
            if row.get("instructions"):
                st.markdown(f"- **指令：** {row.get('instructions')}")
            if row.get("expected_output"):
                st.markdown(f"- **预期输出：** {row.get('expected_output')}")
            details_text = format_agent_log_details(row.get("details"))
            if details_text:
                st.code(details_text, language="json")
            st.divider()


def render_pending_job_notice(job_meta: Dict[str, Any], show_title: bool = False):
    source_name = job_meta.get("source_name") or job_meta.get("report_title") or "未命名论文"
    if show_title:
        st.markdown(f"### {source_name}")

    status = (job_meta.get("status") or "").lower()
    progress_text = job_meta.get("progress_text") or ""
    if status == "failed":
        st.error(progress_text or f"《{source_name}》解析失败。")
        return

    display_text = progress_text or "后台任务正在运行中。"
    st.info(f"《{source_name}》当前状态：{display_text}")
    report_id = job_meta.get("report_id") or job_meta.get("job_id") or ""
    if report_id:
        render_agent_action_logs(st.session_state.get("current_user", ""), report_id, expanded=False)
    st.caption("该解析任务已由后台接管。您可以关闭页面，稍后重新登录查看状态或结果；保持页面打开时系统会自动更新进度。")


@st.cache_data(show_spinner=False, ttl=DB_READ_CACHE_TTL_SECONDS)
def _load_user_report_record_cached(username: str, report_id: str) -> Optional[Dict[str, Any]]:
    row = db_fetch_one(
        """
        SELECT
            j.id AS report_id,
            j.cache_key,
            j.title AS report_title,
            j.pdf_name AS source_name,
            j.created_at AS job_created_at,
            j.updated_at,
            r.created_at AS report_created_at,
            r.report_markdown,
            r.parsed_markdown,
            r.text_agent_output,
            r.vision_output,
            r.images_manifest
        FROM public.analysis_jobs j
        JOIN public.users u ON u.id = j.user_id
        JOIN public.analysis_reports r ON r.job_id = j.id
        WHERE LOWER(u.username) = LOWER(%s) AND j.id = %s
        LIMIT 1
        """,
        (username, report_id),
    )
    if not row:
        return None

    meta = build_report_meta_from_row({
        "report_id": row.get("report_id"),
        "cache_key": row.get("cache_key"),
        "source_name": row.get("source_name"),
        "report_title": row.get("report_title"),
        "created_at": row.get("job_created_at"),
        "updated_at": row.get("updated_at") or row.get("report_created_at") or row.get("job_created_at"),
        "status": "finished",
        "progress_text": "解析完成",
        "has_report": True,
    })
    analysis_result = {
        "source_markdown": row.get("parsed_markdown", "") or "",
        "text_report": row.get("text_agent_output", "") or "",
        "vision_summaries": row.get("vision_output", "") or "",
        "images": normalize_json_field(row.get("images_manifest"), {}),
        "main_report": row.get("report_markdown", "") or "",
    }
    return {"meta": meta, "analysis_result": analysis_result}


def load_user_report_record(username: str, report_id: str) -> Optional[Dict[str, Any]]:
    username = normalize_username(username)
    report_id = (report_id or "").strip()
    if not username or not report_id:
        return None
    ensure_app_storage()
    return _load_user_report_record_cached(username, report_id)


@st.cache_data(show_spinner=False, ttl=DB_READ_CACHE_TTL_SECONDS)
def _get_user_cached_report_cached(username: str, cache_key: str) -> Optional[Dict[str, Any]]:
    row = db_fetch_one(
        """
        SELECT j.id AS report_id
        FROM public.analysis_jobs j
        JOIN public.users u ON u.id = j.user_id
        JOIN public.analysis_reports r ON r.job_id = j.id
        WHERE LOWER(u.username) = LOWER(%s) AND j.cache_key = %s
        ORDER BY COALESCE(j.updated_at, r.created_at, j.created_at) DESC
        LIMIT 1
        """,
        (username, cache_key),
    )
    if not row:
        return None
    payload = _load_user_report_record_cached(username, row.get("report_id", ""))
    if payload and isinstance(payload.get("analysis_result"), dict):
        return payload["analysis_result"]
    return None


def get_user_cached_report(username: str, cache_key: str) -> Optional[Dict[str, Any]]:
    username = normalize_username(username)
    cache_key = (cache_key or "").strip()
    if not username or not cache_key:
        return None
    ensure_app_storage()
    return _get_user_cached_report_cached(username, cache_key)


def shorten_sidebar_label(text: str, max_len: int = 20) -> str:
    value = (text or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "…"


def format_report_history_label(meta: Dict[str, Any]) -> str:
    if not meta:
        return "当前工作区"
    display_name = meta.get("source_name") or meta.get("report_title") or "未命名论文"
    short_name = shorten_sidebar_label(display_name, max_len=18)
    status = (meta.get("status") or "").lower()
    if status in {"queued", "processing"}:
        return f"{short_name}｜正在解析中"
    if status == "failed":
        return f"{short_name}｜解析失败"
    timestamp = (meta.get("updated_at") or meta.get("created_at") or "")[:16]
    return f"{short_name}｜{timestamp}" if timestamp else short_name


def render_auth_ui():
    st.title("学术文献智能工作台")
    st.markdown("请登录研究工作区。系统会为每个账号独立保存文献检索记录与论文精读报告，便于跨会话连续使用。")

    login_tab, register_tab = st.tabs(["登录", "注册"])

    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input("账号", key="login_username")
            login_password = st.text_input("密码", type="password", key="login_password")
            login_submit = st.form_submit_button("登录", use_container_width=True)
            if login_submit:
                ok, result = authenticate_user(login_username, login_password)
                if ok:
                    st.session_state.current_user = result
                    reset_user_workspace_view(result)
                    st.rerun()
                else:
                    st.error(result)

    with register_tab:
        with st.form("register_form"):
            register_username = st.text_input("账号", key="register_username")
            register_password = st.text_input("密码", type="password", key="register_password")
            confirm_password = st.text_input("确认密码", type="password", key="register_password_confirm")
            register_submit = st.form_submit_button("注册并进入系统", use_container_width=True)
            if register_submit:
                if register_password != confirm_password:
                    st.error("两次输入的密码不一致。")
                else:
                    ok, result = register_user(register_username, register_password)
                    if ok:
                        st.session_state.current_user = result
                        reset_user_workspace_view(result)
                        st.rerun()
                    else:
                        st.error(result)

    st.stop()


def render_history_sidebar(username: str):
    history = load_user_report_index(username)
    selector_key = get_history_selector_key(username)
    reset_flag_key = get_history_reset_flag_key(username)
    options = ["__workspace__"] + [item.get("report_id", "") for item in history if item.get("report_id")]
    meta_map = {item.get("report_id", ""): item for item in history if item.get("report_id")}

    if (
        selector_key not in st.session_state
        or st.session_state[selector_key] not in options
        or st.session_state.get(reset_flag_key, False)
    ):
        st.session_state[selector_key] = "__workspace__"
        st.session_state[reset_flag_key] = False

    st.header("精读报告档案")
    selected_id = st.radio(
        "精读报告档案",
        options=options,
        format_func=lambda option_id: "当前工作区" if option_id == "__workspace__" else format_report_history_label(meta_map.get(option_id, {})),
        key=selector_key,
        label_visibility="collapsed",
    )
    st.session_state.selected_history_report_id = None if selected_id == "__workspace__" else selected_id

    if history:
        st.caption("历史报告会长期保留，可在重新登录后继续查看。")
    else:
        st.caption("当前账号暂无精读报告档案。")

    st.divider()
    search_history = load_user_search_index(username)
    search_options = ["__none__"] + [item.get("search_job_id", "") for item in search_history if item.get("search_job_id")]
    search_meta_map = {item.get("search_job_id", ""): item for item in search_history if item.get("search_job_id")}
    search_selector_key = f"search_history_selector_{canonical_username(username)}"
    if selected_id != "__workspace__":
        st.session_state[search_selector_key] = "__none__"
    if search_selector_key not in st.session_state or st.session_state[search_selector_key] not in search_options:
        st.session_state[search_selector_key] = "__none__"
    st.header("文献检索档案")
    selected_search_id = st.radio(
        "文献检索档案",
        options=search_options,
        format_func=lambda option_id: "不打开检索记录" if option_id == "__none__" else format_search_history_label(search_meta_map.get(option_id, {})),
        key=search_selector_key,
        label_visibility="collapsed",
    )
    st.session_state.selected_history_search_id = None if selected_search_id == "__none__" else selected_search_id
    if selected_search_id != "__none__":
        st.session_state.selected_history_report_id = None
    if search_history:
        st.caption("已确认或仍在运行的检索任务会保留在这里，重新登录后可继续查看。")
    else:
        st.caption("当前账号暂无文献检索档案。")

# ==========================================
# 模块 14：论文精读主流程
# ==========================================
def get_pdf_cache_key(pdf_bytes: bytes) -> str:
    """为每篇论文生成稳定缓存键，支持多文件分别缓存。"""
    return hashlib.sha256(ANALYSIS_CACHE_VERSION.encode('utf-8') + pdf_bytes).hexdigest()


def get_or_create_analysis_result(pdf_bytes: bytes, source_name: str) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """读取、提交或轮询单篇论文分析结果。"""
    cache_key = get_pdf_cache_key(pdf_bytes)
    cache_pool = st.session_state.analysis_results
    if cache_key in cache_pool and cache_pool[cache_key] is not None:
        return cache_key, cache_pool[cache_key], "session_cache"

    current_user = normalize_username(st.session_state.get("current_user", ""))
    if not current_user:
        return cache_key, None, "failed"

    cached_report = get_user_cached_report(current_user, cache_key)
    if cached_report is not None:
        cache_pool[cache_key] = cached_report
        return cache_key, cached_report, "history_cache"

    existing_job = get_user_job_by_cache_key(current_user, cache_key)
    if existing_job:
        existing_status = (existing_job.get("status") or "").lower()
        if existing_status in {"queued", "processing"}:
            return cache_key, None, "pending"
        if existing_status == "failed":
            return cache_key, None, "failed"
        if existing_status == "finished":
            return cache_key, None, "processing_finalize"

    try:
        job_meta, should_submit = create_or_reuse_analysis_job(current_user, source_name, cache_key)
        if should_submit:
            submit_analysis_job_to_modal(job_meta["report_id"], source_name, cache_key, pdf_bytes)
            update_analysis_job_status(job_meta["report_id"], "processing", "后台任务已启动，等待离线解析完成")
            return cache_key, None, "submitted"
        status = (job_meta.get("status") or "").lower()
        if status in {"queued", "processing"}:
            return cache_key, None, "pending"
        if status == "finished" and job_meta.get("has_report"):
            cached_report = get_user_cached_report(current_user, cache_key)
            if cached_report is not None:
                cache_pool[cache_key] = cached_report
                return cache_key, cached_report, "history_cache"
        return cache_key, None, "pending"
    except Exception as e:
        existing_job = get_user_job_by_cache_key(current_user, cache_key)
        if existing_job:
            update_analysis_job_status(existing_job["report_id"], "failed", f"后台任务提交失败：{str(e)}")
        cache_pool.pop(cache_key, None)
        return cache_key, None, "failed"


def build_export_filename(source_name: str, suffix: str) -> str:
    base_name = re.sub(r'(?i)\.pdf$', '', (source_name or '').strip())
    base_name = re.sub(r'[\/:*?"<>|]+', '_', base_name).strip() or '论文'
    return f"{base_name}{suffix}"


def collect_pdf_entries(pdf_inputs) -> List[Tuple[str, bytes]]:
    entries: List[Tuple[str, bytes]] = []

    if isinstance(pdf_inputs, bytes):
        entries.append(("论文.pdf", pdf_inputs))
        return entries

    if isinstance(pdf_inputs, tuple) and len(pdf_inputs) == 2:
        name, data = pdf_inputs
        if isinstance(data, (bytes, bytearray)):
            entries.append((str(name) or "论文.pdf", bytes(data)))
        return entries

    for idx, uploaded in enumerate(pdf_inputs or [], start=1):
        if uploaded is None:
            continue
        if isinstance(uploaded, tuple) and len(uploaded) == 2:
            name, data = uploaded
            if isinstance(data, (bytes, bytearray)):
                entries.append((str(name) or f"paper_{idx}.pdf", bytes(data)))
            continue
        if isinstance(uploaded, dict):
            name = str(uploaded.get("name") or f"paper_{idx}.pdf")
            data = uploaded.get("bytes")
            if isinstance(data, (bytes, bytearray)):
                entries.append((name, bytes(data)))
            continue
        paper_name = getattr(uploaded, 'name', f'paper_{idx}.pdf')
        if hasattr(uploaded, 'getvalue'):
            entries.append((paper_name, uploaded.getvalue()))
        elif isinstance(uploaded, (bytes, bytearray)):
            entries.append((paper_name, bytes(uploaded)))
        else:
            entries.append((paper_name, uploaded))

    return entries

def render_comparative_section_summary(analysis_result: Dict[str, Any]):
    agent_state = analysis_result.get("agent_state", {}) or {}
    director_plan = agent_state.get("director_plan", {}) or {}
    comparative_pack = agent_state.get("comparative_memory_pack", {}) or {}
    selected_reports = comparative_pack.get("selected_reports", []) or []
    judge_outputs = comparative_pack.get("problem_judge_outputs", []) or []
    if judge_outputs:
        st.markdown("**同问题审查 Agent 判定结果：**")
        for idx, item in enumerate(judge_outputs, start=1):
            title = item.get("paper_title") or item.get("report_id") or f"候选 {idx}"
            flag = "通过" if item.get("same_problem") else "不通过"
            conf = item.get("confidence") or "low"
            reason = item.get("reason") or ""
            st.markdown(f"{idx}. {title}｜{flag}｜置信度：{conf}｜理由：{reason}")
            
    comparative_enabled = bool(director_plan.get("comparative_section_enabled")) or bool(comparative_pack.get("enabled"))
    comparative_context = str(director_plan.get("comparative_context") or "").strip()

    if not comparative_enabled and not selected_reports:
        return

    with st.expander("查看同问题比较章节来源", expanded=False):
        if comparative_enabled:
            st.success("本次报告已启用“同问题方法比较与综合分析”章节。")
        else:
            st.info("检测到同问题历史论文候选，但本次未正式启用比较章节。")

        if selected_reports:
            st.markdown("**参与候选的历史论文：**")
            for idx, item in enumerate(selected_reports, start=1):
                title = item.get("paper_title") or "未命名历史论文"
                sim = item.get("report_similarity")
                if sim is not None:
                    st.markdown(f"{idx}. {title}（相似度：{float(sim):.3f}）")
                else:
                    st.markdown(f"{idx}. {title}")

        if comparative_context:
            st.markdown("**比较章节使用的摘要材料：**")
            st.code(comparative_context[:4000], language="text")


def render_single_analysis_result(
    analysis_result: Dict[str, Any],
    cache_key: str,
    source_name: str,
    show_paper_title: bool = False,
    status_text: str = "论文深度透视报告已生成！",
):
    if show_paper_title:
        st.markdown(f"### {source_name}")

    st.success(status_text)

    render_comparative_section_summary(analysis_result)

    display_report_md = prepare_report_markdown_for_display(
        analysis_result.get("main_report", ""),
        images_dict=analysis_result.get("images", {}) or {},
        vision_summaries=analysis_result.get("vision_summaries", "") or "",
    )
    render_report_with_images(display_report_md, analysis_result.get("images", {}) or {})

    st.divider()
    st.markdown("### 导出")
    st.download_button(
        label="下载报告原文（Markdown）",
        data=display_report_md,
        file_name=build_export_filename(source_name, "_论文全维度深度透视报告.md"),
        mime="text/markdown",
        use_container_width=True,
        key=f"download_md_{cache_key}",
    )

def render_saved_history_report(username: str, report_id: str):
    job_meta = get_user_job_state(username, report_id)
    if not job_meta:
        st.warning("未找到这条精读报告档案，请重新解析论文。")
        return

    status = (job_meta.get("status") or "").lower()
    source_name = job_meta.get("source_name") or job_meta.get("report_title") or "历史报告"

    if status in {"queued", "processing"}:
        render_pending_job_notice(job_meta, show_title=True)
        st_autorefresh(interval=JOB_STATUS_REFRESH_INTERVAL_MS, key=f"job_status_poll_{report_id}")
        return

    if status == "failed":
        render_pending_job_notice(job_meta, show_title=True)
        return

    payload = load_user_report_record(username, report_id)
    if not payload:
        st.info(f"《{source_name}》的后台任务已完成，正在同步最终报告，请稍候自动刷新。")
        st_autorefresh(interval=JOB_STATUS_REFRESH_INTERVAL_MS, key=f"job_finalize_poll_{report_id}")
        return

    meta = payload.get("meta", {})
    analysis_result = payload.get("analysis_result", {})
    timestamp = meta.get("updated_at") or meta.get("created_at") or "未知时间"

    st.info(f"已载入历史报告：{source_name}（保存时间：{timestamp}）。")
    render_agent_action_logs(username, report_id, expanded=False)
    render_single_analysis_result(
        analysis_result=analysis_result,
        cache_key=meta.get("cache_key", report_id),
        source_name=source_name,
        show_paper_title=True,
        status_text="历史报告已载入，无需重新解析。",
    )


def render_batch_status_overview(batch_rows: List[Dict[str, Any]]):
    st.markdown("### 批量解析进度")
    for row in batch_rows:
        idx = row.get("index")
        source_name = row.get("source_name") or "未命名论文"
        status = (row.get("status") or "").lower()
        progress_text = row.get("progress_text") or ""
        if status == "finished":
            icon = "✅"
            text = progress_text or "解析完成"
        elif status in {"queued", "processing"}:
            icon = "⏳"
            text = progress_text or "正在解析中"
        else:
            icon = "❌"
            text = progress_text or "解析失败"
        st.markdown(f"**{icon} 第 {idx} 篇《{source_name}》：** {text}")


def render_analysis_ui(pdf_inputs):
    """
    上传论文后的主工作流：
    - 支持单篇 PDF 分析
    - 支持多篇 PDF 同时上传，并分别生成各自报告
    - 任务会立即写入 analysis_jobs，并交由后台解析工作流继续执行
    - 多篇 PDF 场景下，全部完成后再统一展示所有报告
    """
    entries = collect_pdf_entries(pdf_inputs)
    if not entries:
        return

    current_user = normalize_username(st.session_state.get("current_user", ""))
    multi_mode = len(entries) > 1
    should_poll = False
    batch_rows: List[Dict[str, Any]] = []
    ready_reports: List[Dict[str, Any]] = []

    for idx, (paper_name, pdf_bytes) in enumerate(entries, start=1):
        cache_key, analysis_result, result_source = get_or_create_analysis_result(pdf_bytes, paper_name)
        job_meta = get_user_job_by_cache_key(current_user, cache_key) if current_user else None

        row = {
            "index": idx,
            "source_name": paper_name,
            "cache_key": cache_key,
            "status": "processing",
            "progress_text": "任务正在初始化中，请稍候。",
        }

        if analysis_result is not None:
            row.update({"status": "finished", "progress_text": "解析完成"})
            ready_reports.append({
                "index": idx,
                "source_name": paper_name,
                "cache_key": cache_key,
                "analysis_result": analysis_result,
                "result_source": result_source,
            })
        else:
            if job_meta:
                row.update({
                    "report_id": job_meta.get("report_id") or job_meta.get("job_id") or "",
                    "status": job_meta.get("status") or "processing",
                    "progress_text": job_meta.get("progress_text") or row["progress_text"],
                })
            elif result_source == "failed":
                row.update({"status": "failed", "progress_text": "后台任务提交或执行失败，请稍后重试。"})
            elif result_source == "submitted":
                row.update({"status": "processing", "progress_text": "后台任务已创建，正在等待离线解析。"})
            elif result_source == "processing_finalize":
                row.update({"status": "processing", "progress_text": "任务已完成，正在同步最终报告。"})

        if row["status"] in {"queued", "processing"}:
            should_poll = True

        batch_rows.append(row)

    if multi_mode:
        render_batch_status_overview(batch_rows)
        pending_exists = any((row.get("status") or "").lower() in {"queued", "processing"} for row in batch_rows)
        if pending_exists:
            finished_count = sum(1 for row in batch_rows if (row.get("status") or "").lower() == "finished")
            if finished_count:
                st.info("已有部分论文解析完成。为避免界面混乱，系统会在全部任务完成后统一展示所有 PDF 的解析报告。")
            if should_poll:
                st_autorefresh(interval=JOB_STATUS_REFRESH_INTERVAL_MS, key="pending_analysis_refresh")
                st.caption("后台任务正在继续运行。您可以关闭页面，稍后重新登录查看；若保持当前页面打开，系统会每 3 分钟自动刷新一次。")
        else:
            st.divider()
            st.markdown("### 全部 PDF 解析结果")
            rendered_sections = 0
            total_sections = len(ready_reports) + sum(1 for row in batch_rows if (row.get("status") or "").lower() == "failed")
            for report_item in ready_reports:
                rendered_sections += 1
                st.markdown(f"## 第 {report_item['index']} 篇论文")
                if report_item["result_source"] == "history_cache":
                    st.info(f"检测到当前账号下已存在《{report_item['source_name']}》的历史报告，已直接读取，无需重新解析。")
                render_single_analysis_result(
                    analysis_result=report_item["analysis_result"],
                    cache_key=report_item["cache_key"],
                    source_name=report_item["source_name"],
                    show_paper_title=False,
                )
                if rendered_sections < total_sections:
                    st.divider()
            for row in batch_rows:
                if (row.get("status") or "").lower() != "failed":
                    continue
                rendered_sections += 1
                st.markdown(f"## 第 {row['index']} 篇论文")
                st.error(f"《{row['source_name']}》解析失败：{row.get('progress_text') or '未知错误'}")
                if rendered_sections < total_sections:
                    st.divider()
    else:
        report_item = ready_reports[0] if ready_reports else None
        row = batch_rows[0]
        if report_item:
            if report_item["result_source"] == "history_cache":
                st.info(f"检测到当前账号下已存在《{report_item['source_name']}》的历史报告，已直接读取，无需重新解析。")
            render_single_analysis_result(
                analysis_result=report_item["analysis_result"],
                cache_key=report_item["cache_key"],
                source_name=report_item["source_name"],
                show_paper_title=False,
            )
        else:
            render_pending_job_notice(row, show_title=False)
            if should_poll:
                st_autorefresh(interval=JOB_STATUS_REFRESH_INTERVAL_MS, key="pending_analysis_refresh")
                st.caption("后台任务正在继续运行。您可以关闭页面，稍后重新登录查看；若保持当前页面打开，系统会每 3 分钟自动刷新一次。")

    st.divider()
    if st.button("返回当前工作区", type="primary", key="start_fresh_workspace_after_analysis"):
        start_fresh_workspace(st.session_state.get("current_user", ""))
        st.rerun()


# ==========================================
# 模块 15：状态初始化
# ==========================================
try:
    ensure_app_storage()
except Exception as e:
    st.error(str(e))
    st.stop()

if "current_user" not in st.session_state:
    st.session_state.current_user = ""
if "selected_history_report_id" not in st.session_state:
    st.session_state.selected_history_report_id = None
if "selected_history_search_id" not in st.session_state:
    st.session_state.selected_history_search_id = None
if "active_search_job_id" not in st.session_state:
    st.session_state.active_search_job_id = ""
if "app_state" not in st.session_state:
    st.session_state.app_state = "IDLE"
if "final_result" not in st.session_state:
    st.session_state.final_result = ""
if "has_provided_feedback" not in st.session_state:
    st.session_state.has_provided_feedback = False
if "ui_logs" not in st.session_state:
    st.session_state.ui_logs = []
if "search_topic" not in st.session_state:
    st.session_state.search_topic = ""
if "search_requirements" not in st.session_state:
    st.session_state.search_requirements = ""
if "search_preprint_rule" not in st.session_state:
    st.session_state.search_preprint_rule = "排除预印本 (仅限正规期刊/会议)"
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "feedback_start_time" not in st.session_state:
    st.session_state.feedback_start_time = None
if "sidebar_direct_entries" not in st.session_state:
    st.session_state.sidebar_direct_entries = []
if "bottom_direct_entries" not in st.session_state:
    st.session_state.bottom_direct_entries = []

if not st.session_state.current_user:
    render_auth_ui()


# ==========================================
# 模块 16：前端 UI
# ==========================================
st.title("学术文献智能工作台")
st.markdown("面向学术选题、文献筛选与论文精读的统一多 Agent 工作区。系统支持后台检索、离线任务续跑、历史档案留存与结构化精读报告生成。")

with st.sidebar:
    st.success(f"当前账号：{st.session_state.current_user}")
    if st.button("退出登录", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.divider()
    render_history_sidebar(st.session_state.current_user)

    st.divider()
    st.header("研究检索配置")

    user_topic = st.text_input("研究主题", value="")

    user_requirements = st.text_area(
        "筛选约束与偏好",
        value="",
        placeholder="例如：限定任务类型、方法路线、应用场景、数据集、发表渠道或排除条件。",
        help="约束越清晰，后端检索 Agent 越容易稳定筛选出高相关文献。",
    )

    allow_preprint = st.radio(
        "收录范围",
        ("排除预印本 (仅限正规期刊/会议)", "接受预印本 (如 arXiv)"),
    )

    start_button = st.button("启动文献检索任务", type="primary", use_container_width=True)

    st.divider()

    st.header("论文精读入口")
    sidebar_pdf = st.file_uploader(
        "上传 PDF 进行结构化精读",
        type="pdf",
        key="sb_pdf",
        help="跳过检索流程，直接为已有论文生成结构化精读报告，并写入当前账号档案。",
        accept_multiple_files=True,
    )

    start_analyze_button = st.button(
        "启动深度解析",
        type="primary",
        key="start_analyze_btn",
        use_container_width=True,
        disabled=not sidebar_pdf,
    )


# ==========================================
# 模块 17：业务路由与后端检索提交
# ==========================================
if start_analyze_button and sidebar_pdf:
    reset_user_workspace_view(st.session_state.current_user)
    st.session_state.app_state = "IDLE"
    st.session_state.bottom_direct_entries = []
    st.session_state.sidebar_direct_entries = collect_pdf_entries(sidebar_pdf)
    st.rerun()

if start_button:
    if not user_topic:
        st.warning("请填写研究主题！")
    else:
        reset_user_workspace_view(st.session_state.current_user)
        st.session_state.sidebar_direct_entries = []
        st.session_state.bottom_direct_entries = []
        st.session_state.search_topic = user_topic
        st.session_state.search_requirements = user_requirements
        st.session_state.search_preprint_rule = allow_preprint
        st.session_state.final_result = ""
        st.session_state.ui_logs = []
        st.session_state.has_provided_feedback = False
        st.session_state.feedback_start_time = None
        search_submitted = False
        with st.spinner("正在创建后台检索任务，并交由统一 Director 调度……"):
            try:
                search_job = create_paper_search_job(
                    username=st.session_state.current_user,
                    user_topic=user_topic,
                    user_requirements=user_requirements,
                    preprint_rule=allow_preprint,
                )
                submit_paper_search_to_modal(search_job.get("search_job_id", ""))
                update_paper_search_job_status(search_job.get("search_job_id", ""), "processing", "后台检索任务已提交，等待 Agent 完成候选文献筛选")
                st.session_state.active_search_job_id = search_job.get("search_job_id", "")
                st.session_state.current_search_job_id = search_job.get("search_job_id", "")
                st.session_state.app_state = "SEARCH_RUNNING"
                search_submitted = True
            except Exception as e:
                st.session_state.app_state = "IDLE"
                st.error(f"后台文献检索提交失败：{str(e)}")
        if search_submitted:
            st.rerun()

if st.session_state.sidebar_direct_entries:
    st.markdown("---")
    st.info("已进入论文精读流程，正在启动结构化解析任务……")
    render_analysis_ui(st.session_state.sidebar_direct_entries)
    st.stop()

if st.session_state.app_state == "SEARCH_RUNNING" and st.session_state.active_search_job_id:
    st.markdown("---")
    current_search_state = poll_active_paper_search_job(st.session_state.current_user, st.session_state.active_search_job_id)
    if current_search_state and st.session_state.app_state == "SEARCH_RUNNING":
        active_search_meta = current_search_state.get("meta", {}) or {}
        render_pending_search_notice(active_search_meta)
        st_autorefresh(interval=JOB_STATUS_REFRESH_INTERVAL_MS, key=f"active_search_refresh_{st.session_state.active_search_job_id}")
        st.stop()
    st.rerun()

if (
    st.session_state.selected_history_search_id
    and st.session_state.app_state == "IDLE"
    and not st.session_state.bottom_direct_entries
):
    st.markdown("---")
    render_saved_search_record(st.session_state.current_user, st.session_state.selected_history_search_id)
    st.stop()

if (
    st.session_state.selected_history_report_id
    and st.session_state.app_state == "IDLE"
    and not st.session_state.bottom_direct_entries
):
    st.markdown("---")
    render_saved_history_report(st.session_state.current_user, st.session_state.selected_history_report_id)
    st.stop()

if st.session_state.app_state == "IDLE":
    st.markdown("""
### 当前工作区

从左侧选择你要进行的任务。你可以先输入研究主题进行文献检索，也可以直接上传已有论文进行精读。任务提交后，即使离开页面，稍后重新登录也可以继续查看结果。

**文献检索**

填写研究主题和筛选要求后，系统会为你查找相关论文，并整理出推荐结果。  
结果生成后，你可以选择保存当前结果，也可以继续修改要求重新检索。

**论文精读**

上传论文 PDF 后，系统会生成一份精读报告，帮助你快速理解论文的研究问题、方法设计、实验结果、图表内容和主要结论。

**历史档案**

左侧可以查看你之前保存的文献检索结果和论文精读报告。
""")

if st.session_state.app_state != "IDLE":
    st.markdown("### 检索 Agent 执行轨迹")
    for log in st.session_state.ui_logs:
        with st.expander(log["title"], expanded=False):
            st.markdown(log["content"])

# 论文搜索 Agent 已完全迁移到后端；前端不再执行本地 Agent 循环。
if st.session_state.app_state == "WAITING_FEEDBACK":
    if st.session_state.current_search_job_id:
        try:
            current_meta = get_user_search_job_state(st.session_state.current_user, st.session_state.current_search_job_id)
            if current_meta and current_meta.get("is_final"):
                st.session_state.app_state = "COMPLETED"
                st.session_state.feedback_start_time = None
                st.rerun()
        except Exception:
            pass

    if st.session_state.feedback_start_time:
        elapsed_time = time.time() - st.session_state.feedback_start_time
        remaining_time = 1800 - elapsed_time

        if remaining_time <= 0:
            try:
                finalize_paper_search_via_modal(st.session_state.current_search_job_id)
            except Exception as e:
                st.warning(f"自动确认归档失败：{str(e)}")
            st.session_state.app_state = "COMPLETED"
            st.rerun()
        st_autorefresh(interval=JOB_STATUS_REFRESH_INTERVAL_MS, key="feedback_timer")
        mins_left = int(remaining_time // 60)
        st.caption(f"若无进一步操作，后端将在约 {mins_left} 分钟后自动确认当前结果并归档；关闭页面不会影响该计时。")

    st.markdown("### 候选文献组合")
    st.write("请审阅当前候选文献组合。确认后，本轮结果才会写入最终去重库；若继续修正，本轮结果不会作为最终推荐保存。")

    with st.container(border=True):
        st.markdown(st.session_state.final_result)

    st.divider()
    st.markdown("#### 是否确认当前候选组合？")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("确认当前结果并归档", use_container_width=True):
            try:
                finalize_paper_search_via_modal(st.session_state.current_search_job_id)
                st.session_state.app_state = "COMPLETED"
                st.rerun()
            except Exception as e:
                st.error(f"结果确认归档失败：{str(e)}")

    with col2:
        with st.popover("继续修正检索条件", use_container_width=True):
            new_req = st.text_area("请说明需要修正的方向或需要排除的结果：")
            if st.button("提交修正要求并重新检索"):
                if new_req.strip():
                    with st.spinner("正在创建修正后的后台检索任务……"):
                        try:
                            previous_search_job_id = st.session_state.current_search_job_id
                            search_job = create_paper_search_job(
                                username=st.session_state.current_user,
                                user_topic=st.session_state.search_topic,
                                user_requirements=st.session_state.search_requirements,
                                preprint_rule=st.session_state.search_preprint_rule,
                                feedback=new_req.strip(),
                                previous_result=st.session_state.final_result,
                            )
                            mark_paper_search_job_superseded(previous_search_job_id, search_job.get("search_job_id", ""))
                            submit_paper_search_to_modal(search_job.get("search_job_id", ""))
                            update_paper_search_job_status(search_job.get("search_job_id", ""), "processing", "修正检索任务已提交，等待 Agent 完成新一轮筛选")
                            st.session_state.active_search_job_id = search_job.get("search_job_id", "")
                            st.session_state.current_search_job_id = search_job.get("search_job_id", "")
                            st.session_state.has_provided_feedback = True
                            st.session_state.app_state = "SEARCH_RUNNING"
                            st.session_state.feedback_start_time = None
                        except Exception as e:
                            st.error(f"修正检索提交失败：{str(e)}")
                    st.rerun()

elif st.session_state.app_state == "COMPLETED":
    st.success("文献检索任务已确认归档。")
    if st.session_state.has_provided_feedback == False and st.session_state.feedback_start_time:
        elapsed = time.time() - st.session_state.feedback_start_time
        if elapsed > 1800:
            st.warning("提示：由于超过 30 分钟未响应，系统已为您自动确认最终结果。")
    st.markdown("### 最终确认的六篇候选文献")
    with st.container(border=True):
        st.markdown(st.session_state.final_result)

    st.divider()
    st.header("论文精读工作流")
    st.info("从上方选定并下载任意一篇或多篇论文的 PDF，在此上传，系统将分别生成完整 7 节精读报告，并自动存入当前账号的精读报告档案。")

    uploaded_pdf = st.file_uploader("上传 PDF 生成精读报告", type="pdf", key="bottom_pdf", accept_multiple_files=True)
    bottom_start_btn = st.button(
        "启动深度解析",
        type="primary",
        disabled=not uploaded_pdf,
        use_container_width=True,
    )
    if bottom_start_btn and uploaded_pdf:
        reset_user_workspace_view(st.session_state.current_user)
        st.session_state.bottom_direct_entries = collect_pdf_entries(uploaded_pdf)
        st.rerun()

    if st.session_state.bottom_direct_entries:
        st.markdown("---")
        render_analysis_ui(st.session_state.bottom_direct_entries)

    if st.button("返回工作区并开启新任务", type="primary"):
        current_user = st.session_state.get("current_user", "")
        st.session_state.clear()
        if current_user:
            st.session_state.current_user = current_user
        st.rerun()
