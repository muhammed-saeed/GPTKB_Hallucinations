#!/usr/bin/env python3
"""
Test the EXACT flow that the crawler uses for elicitation.
This mimics what happens at line 383 in crawler_simple.py
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

print("Testing EXACT Crawler Elicitation Flow\n")
print("=" * 70)

# Import what the crawler imports
from llm.factory import make_llm_from_config, ModelConfig
from settings import ELICIT_SCHEMA_BASE

print("\n[Step 1] Setting up like the crawler does...")

# Create config like the crawler does
el_cfg = ModelConfig(
    provider="deepseek",
    model="deepseek-chat",
    api_key_env="DEEPSEEK_API_KEY",
    temperature=0.2,
    top_p=1.0,
    max_tokens=2048,
)

print(f"   Config: provider={el_cfg.provider}, model={el_cfg.model}")
print(f"   Temperature: {el_cfg.temperature}")
print(f"   Max tokens: {el_cfg.max_tokens}")

# Create LLM like the crawler does
el_llm = make_llm_from_config(el_cfg)
print(f"   LLM created: {type(el_llm)}")

# Get the schema like the crawler does
print(f"\n[Step 2] Preparing messages and schema...")
print(f"   ELICIT_SCHEMA_BASE keys: {ELICIT_SCHEMA_BASE.keys()}")

# Simple test message
subject = "Test Subject"
user_prompt = f"""You are a knowledge base expert.
Return facts about {subject}.

Output ONLY JSON with "facts" array."""

el_messages = [
    {"role": "system", "content": "Return JSON only."},
    {"role": "user", "content": user_prompt},
]

el_schema = ELICIT_SCHEMA_BASE

print(f"   Messages: {len(el_messages)} items")
print(f"   Schema: {list(el_schema.keys())}")

# Call EXACTLY like the crawler does at line 383
print(f"\n[Step 3] Calling elicitation (line 383 in crawler)...")
print(f"   Calling: el_llm(el_messages, json_schema=el_schema)")

try:
    resp = el_llm(el_messages, json_schema=el_schema)
    
    print(f"\n[Step 4] Response received...")
    print(f"   Response type: {type(resp)}")
    print(f"   Response keys: {list(resp.keys())}")
    
    # Check what we got
    if "_raw" in resp:
        print(f"\n   ✗ PROBLEM: Got _raw!")
        print(f"   _raw length: {len(resp['_raw'])}")
        print(f"   _raw (first 200): {resp['_raw'][:200]}")
        
        # Now test if _loose_json_from_raw can parse it
        print(f"\n[Step 5] Testing _loose_json_from_raw() on this _raw...")
        from crawler_simple import _loose_json_from_raw
        
        loose = _loose_json_from_raw(resp["_raw"])
        print(f"   Loose result: {list(loose.keys())}")
        print(f"   Facts: {len(loose.get('facts', []))}")
        
        if loose.get('facts'):
            print(f"\n   ✓ _loose_json_from_raw() CAN parse it!")
        else:
            print(f"\n   ✗ _loose_json_from_raw() CANNOT parse it!")
            
    elif "facts" in resp:
        print(f"\n   ✓ Got facts directly!")
        print(f"   Facts count: {len(resp['facts'])}")
    else:
        print(f"\n   ? Unknown response: {resp}")
        
except Exception as e:
    print(f"\n   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)