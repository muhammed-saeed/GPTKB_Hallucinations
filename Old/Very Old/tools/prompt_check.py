# tools/prompt_check.py
import os, sys, argparse

# Ensure we import the project root's prompter_parser.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from prompter_parser import get_prompt_template  # now the real one in project root

def main():
    ap = argparse.ArgumentParser(description="Render a prompt from your templates.")
    ap.add_argument("--strategy", required=True, choices=["baseline", "icl", "dont_know", "calibrate"])
    ap.add_argument("--task", required=True, choices=["elicitation", "ner"])
    ap.add_argument("--domain", required=True, choices=["general", "topic"])
    ap.add_argument("--topic", default=None)
    ap.add_argument("--subject", default="Vannevar Bush")
    ap.add_argument("--max-facts-hint", default="20")
    args = ap.parse_args()

    print(f"\n== prompt_check ==\nstrategy={args.strategy} task={args.task} domain={args.domain} topic={args.topic}")

    tpl = get_prompt_template(args.strategy, args.task, domain=args.domain, topic=args.topic)

    if args.task == "elicitation":
        txt = tpl.render(subject_name=args.subject, max_facts_hint=args.max_facts_hint)
    else:
        # simple demo lines for NER
        txt = tpl.render(lines="Alan Turing\nMIT\nNew York City")

    print("\n--- RENDERED PROMPT ---\n")
    print(txt)
    print("\n-----------------------\n")

if __name__ == "__main__":
    main()
