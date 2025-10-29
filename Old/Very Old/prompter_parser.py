# # prompter_parser.py
# import os
# from jinja2 import Environment, FileSystemLoader, TemplateNotFound, StrictUndefined, Undefined

# _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# _PROMPTS_DIR = os.path.join(_THIS_DIR, "prompts")

# # In debug, fail loudly on missing template vars.
# _DEBUG = bool(os.getenv("DEBUG_PROMPTS"))
# _ENV = Environment(
#     loader=FileSystemLoader(_PROMPTS_DIR),
#     autoescape=False,
#     undefined=StrictUndefined if _DEBUG else Undefined,
# )

# def _try(path: str):
#     try:
#         return _ENV.get_template(path)
#     except TemplateNotFound:
#         return None

# def get_prompt_template(
#     strategy: str,
#     task: str,                    # "elicitation" | "ner"
#     domain: str = "general",      # "general" | "topic"
#     topic: str | None = None,     # e.g., "entertainment"
# ):
#     """
#     Resolution order (first hit wins):
#       1) topic/<topic>/<strategy>/<task>.j2   (when domain=="topic" and topic provided)
#       2) topic/<topic>/baseline/<task>.j2
#       3) general/<strategy>/<task>.j2
#       4) general/baseline/<task>.j2
#       5) baseline/<task>.j2                   (legacy)
#     """
#     candidates = []
#     if domain == "topic" and topic:
#         candidates += [
#             f"topic/{topic}/{strategy}/{task}.j2",
#             f"topic/{topic}/baseline/{task}.j2",
#         ]
#     candidates += [
#         f"general/{strategy}/{task}.j2",
#         f"general/baseline/{task}.j2",
#         f"baseline/{task}.j2",
#     ]
#     for c in candidates:
#         tpl = _try(c)
#         if tpl:
#             if _DEBUG:
#                 print(f"[prompts] using: {c} (root={_PROMPTS_DIR})")
#             return tpl
#     search_roots = "\n - ".join(candidates)
#     raise FileNotFoundError(
#         f"No prompt found for domain='{domain}', topic='{topic}', strategy='{strategy}', task='{task}'. "
#         f"Searched:\n - {search_roots}\nRoot: {_PROMPTS_DIR}"
#     )


# prompter_parser.py
import os
from jinja2 import Environment, FileSystemLoader, TemplateNotFound, StrictUndefined, Undefined

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROMPTS_DIR = os.path.join(_THIS_DIR, "prompts")

# In debug, fail loudly on missing template vars and show chosen template path.
_DEBUG = bool(os.getenv("DEBUG_PROMPTS"))
_ENV = Environment(
    loader=FileSystemLoader(_PROMPTS_DIR),
    autoescape=False,
    undefined=StrictUndefined if _DEBUG else Undefined,
)

# Map common variants to your on-disk folder names
# (case-insensitive input → exact folder spelling you use)
_STRATEGY_ALIASES = {
    "baseline": "baseline",
    "icl": "ICL",
    "dont_know": "dont_know",
    "dont-know": "dont_know",
    "dontknow": "dont_know",
    "calibrate": "calibration",
    "calibration": "calibration",
}

def _normalize_strategy(name: str) -> str:
    if not name:
        return "baseline"
    key = name.strip().lower()
    return _STRATEGY_ALIASES.get(key, name)  # fallback to raw name if unknown

def _try(path: str):
    try:
        return _ENV.get_template(path)
    except TemplateNotFound:
        return None

def get_prompt_template(
    strategy: str,
    task: str,                    # "elicitation" | "ner" | (anything else you add)
    domain: str = "general",      # "general" | "topic"
    topic: str | None = None,     # e.g., "entertainment"
):
    """
    Resolution order (first hit wins):
      1) topic/<topic>/<strategy>/<task>.j2   (when domain=="topic" and topic provided)
      2) topic/<topic>/baseline/<task>.j2
      3) general/<strategy>/<task>.j2
      4) general/baseline/<task>.j2
      5) baseline/<task>.j2                   (legacy)
    """
    norm_strategy = _normalize_strategy(strategy)

    candidates = []
    if domain == "topic" and topic:
        candidates += [
            f"topic/{topic}/{norm_strategy}/{task}.j2",
            f"topic/{topic}/baseline/{task}.j2",
        ]
    candidates += [
        f"general/{norm_strategy}/{task}.j2",
        f"general/baseline/{task}.j2",
        f"baseline/{task}.j2",
    ]

    for c in candidates:
        tpl = _try(c)
        if tpl:
            if _DEBUG:
                print(f"[prompts] using: {c} (root={_PROMPTS_DIR})")
            return tpl

    search_roots = "\n - ".join(candidates)
    raise FileNotFoundError(
        f"No prompt found for domain='{domain}', topic='{topic}', "
        f"strategy='{strategy}' -> '{norm_strategy}', task='{task}'. "
        f"Searched:\n - {search_roots}\nRoot: {_PROMPTS_DIR}"
    )
