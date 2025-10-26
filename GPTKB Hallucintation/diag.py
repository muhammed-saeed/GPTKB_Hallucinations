import os, json, traceback
from dotenv import load_dotenv
load_dotenv()

print("=== ENV CHECK ===")
print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
print("DEEPSEEK_API_KEY set:", bool(os.getenv("DEEPSEEK_API_KEY")))
print("REPLICATE_API_TOKEN set:", bool(os.getenv("REPLICATE_API_TOKEN")))
print("OPENAI_BASE_URL:", os.getenv("OPENAI_BASE_URL"))

print("\n=== MODEL CALL TEST ===")
from settings import settings, ELICIT_SCHEMA_BASE
from llm.factory import make_llm_from_config

cfg = settings.MODELS[settings.ELICIT_MODEL_KEY].model_copy(deep=True)
llm = make_llm_from_config(cfg)

messages = [
    {"role":"system","content":"Return JSON only."},
    {"role":"user","content": 'Return {"facts":[{"subject":"HealthCheck","predicate":"instanceOf","object":"test"}]}'}
]
try:
    out = llm(messages, json_schema=ELICIT_SCHEMA_BASE)
    print("CALL OK:\n", json.dumps(out, indent=2, ensure_ascii=False))
except Exception:
    print("CALL FAILED:\n", traceback.format_exc())
