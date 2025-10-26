import os, sys, json, textwrap
from dotenv import load_dotenv, find_dotenv

def mask(v: str | None, keep=4):
    if not v:
        return None
    v = v.strip()
    if len(v) <= keep:
        return "*" * len(v)
    return f"{v[:2]}…{v[-keep:]} (len={len(v)})"

def main():
    print("== env_check ==\n")

    # Basic process/paths
    print(f"CWD:         {os.getcwd()}")
    print(f"__file__dir: {os.path.dirname(os.path.abspath(__file__))}")
    print("\nsys.path (top 5):")
    for i, p in enumerate(sys.path[:5]):
        print(f"  [{i}] {p}")

    # Locate and load .env
    env_path = find_dotenv(usecwd=True)
    print(f"\nfind_dotenv(usecwd=True): {env_path or '(not found)'}")
    loaded = load_dotenv(env_path if env_path else None)
    print(f"load_dotenv: {loaded}")

    # Show key envs (masked)
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
        "DEEPSEEK_BASE_URL": os.getenv("DEEPSEEK_BASE_URL"),
        "REPLICATE_API_TOKEN": os.getenv("REPLICATE_API_TOKEN"),
    }
    print("\nEnvironment (masked):")
    for k, v in keys.items():
        print(f"  {k} = {mask(v) if 'KEY' in k or 'TOKEN' in k else (v or None)}")

    # Optional: verify prompts root is accessible
    try:
        # ensure project root import
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from prompter_parser import get_prompt_template
        tpl = get_prompt_template("baseline", "elicitation", domain="general", topic=None)
        print("\nPrompts: OK — template resolved.")
        # Print where the Jinja Environment is rooted by triggering DEBUG
        os.environ["DEBUG_PROMPTS"] = "1"
        _ = get_prompt_template("baseline", "elicitation", domain="general", topic=None)
    except Exception as e:
        print("\nPrompts: FAILED to resolve a template:")
        print(textwrap.indent(str(e), "  "))

if __name__ == "__main__":
    main()
