#!/usr/bin/env python3
"""
Test the crawler's integration with DeepSeek client.
Run from project root!
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

print("Testing Crawler Integration with DeepSeek\n")
print("=" * 60)
print(f"Current directory: {os.getcwd()}")
print(f"Project root: {os.path.dirname(os.path.abspath(__file__))}")

# Step 1: Import and test the client directly
print("\n[Step 1] Testing DeepSeekClient directly...")
try:
    from llm.deepseek_client import DeepSeekClient
    
    client = DeepSeekClient(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )
    
    messages = [
        {"role": "user", "content": 'Return JSON: {"test": "works"}'}
    ]
    
    schema = {"type": "object", "properties": {"test": {"type": "string"}}}
    
    print(f"   Calling client with json_schema: {schema is not None}")
    result = client(messages, json_schema=schema)
    
    print(f"   Result type: {type(result)}")
    print(f"   Result keys: {list(result.keys())}")
    print(f"   Has 'test' key? {'test' in result}")
    print(f"   Has '_raw' key? {'_raw' in result}")
    
    if '_raw' in result:
        print(f"   ✗ PROBLEM: Got _raw instead of parsed JSON")
        print(f"   _raw content (first 100): {str(result['_raw'])[:100]}")
    elif 'test' in result:
        print(f"   ✓ SUCCESS: Client returned parsed JSON")
    else:
        print(f"   ? UNKNOWN: Result has keys: {result.keys()}")
        
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Step 2: Test factory
print("\n[Step 2] Testing factory.make_llm_from_config()...")
try:
    from llm.factory import make_llm_from_config, ModelConfig
    
    config = ModelConfig(
        provider="deepseek",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
    )
    
    print(f"   Config created: provider={config.provider}, model={config.model}")
    
    llm = make_llm_from_config(config)
    
    print(f"   LLM wrapper created: {type(llm)}")
    
    messages = [
        {"role": "user", "content": 'Return JSON: {"factory": "test"}'}
    ]
    
    schema = {"type": "object"}
    
    print(f"   Calling llm(messages, json_schema={schema})")
    result = llm(messages, json_schema=schema)
    
    print(f"   Result type: {type(result)}")
    print(f"   Result keys: {list(result.keys())}")
    print(f"   Has '_raw' key? {'_raw' in result}")
    
    if '_raw' in result:
        print(f"   ✗ PROBLEM: Factory returned _raw")
        print(f"   Content (first 100): {str(result['_raw'])[:100]}")
    else:
        print(f"   ✓ SUCCESS: Factory returned parsed JSON")
        
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Test _loose_json_from_raw
print("\n[Step 3] Testing _loose_json_from_raw()...")
try:
    from crawler_simple import _loose_json_from_raw
    
    # Test with escaped JSON like from _raw
    test_raw = r'{\n    "facts": [{"subject": "test", "predicate": "is", "object": "test"}]\n}'
    
    print(f"   Input (raw): {test_raw[:80]}...")
    result = _loose_json_from_raw(test_raw)
    
    print(f"   Result type: {type(result)}")
    print(f"   Result keys: {list(result.keys())}")
    if result:
        print(f"   Facts count: {len(result.get('facts', []))}")
    
    if result.get('facts'):
        print(f"   ✓ SUCCESS: _loose_json_from_raw() works")
    else:
        print(f"   ✗ PROBLEM: _loose_json_from_raw() returned empty")
        print(f"   Result: {result}")
        
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("\nSummary:")
print("  If all 3 pass: Your setup is working!")
print("  If Step 1 fails: DeepSeek client issue")
print("  If Step 2 fails: Factory issue")
print("  If Step 3 fails: _loose_json_from_raw() issue")