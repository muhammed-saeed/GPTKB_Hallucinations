# prompter_parser.py
import os
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROMPTS_DIR = os.path.join(_THIS_DIR, "prompts")
_ENV = Environment(loader=FileSystemLoader(_PROMPTS_DIR), autoescape=False)

def _try(path: str):
    try:
        return _ENV.get_template(path)
    except TemplateNotFound:
        return None

def get_prompt_template(
    strategy: str,
    task: str,                    # "elicitation" | "ner"
    domain: str = "general",      # "general" | "topic"
    topic: str | None = None,     # e.g., "entertainment"
):
    """
    Resolution order (first hit wins):
      1) topic/<topic>/<strategy>/<task>.j2   (when domain=="topic" and topic provided)
      2) topic/<topic>/baseline/<task>.j2     (topic fallback)
      3) general/<strategy>/<task>.j2
      4) general/baseline/<task>.j2
      5) baseline/<task>.j2                   (legacy)
    """
    candidates = []
    if domain == "topic" and topic:
        candidates += [
            f"topic/{topic}/{strategy}/{task}.j2",
            f"topic/{topic}/baseline/{task}.j2",
        ]
    candidates += [
        f"general/{strategy}/{task}.j2",
        f"general/baseline/{task}.j2",
        f"baseline/{task}.j2",     # legacy flat
    ]
    for c in candidates:
        tpl = _try(c)
        if tpl:
            if os.getenv("DEBUG_PROMPTS"):
                print(f"[prompts] using: {c} (root={_PROMPTS_DIR})")
            return tpl
    search_roots = "\n - ".join(candidates)
    raise FileNotFoundError(
        f"No prompt found for domain='{domain}', topic='{topic}', strategy='{strategy}', task='{task}'. "
        f"Searched:\n - {search_roots}\nRoot: {_PROMPTS_DIR}"
    )
