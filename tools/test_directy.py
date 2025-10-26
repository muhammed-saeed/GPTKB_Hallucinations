#!/usr/bin/env python3
"""
Test DeepSeek API directly to see what it actually returns.
"""

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    print("ERROR: DEEPSEEK_API_KEY not set")
    exit(1)

print("Testing DeepSeek API directly...\n")

url = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

body = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "user", "content": 'Return ONLY this JSON: {"test": "value"}'}
    ],
    "temperature": 0.2,
    "top_p": 1.0,
    "max_tokens": 500,
    "response_format": {"type": "json_object"}
}

print(f"1. Making request to: {url}")
print(f"2. With body keys: {body.keys()}")

response = requests.post(url, headers=headers, json=body, timeout=90.0)

print(f"\n3. Response status: {response.status_code}")

if response.status_code != 200:
    print(f"   ERROR: {response.text}")
    exit(1)

data = response.json()

print(f"\n4. Response data keys: {data.keys()}")
print(f"5. Number of choices: {len(data['choices'])}")

content = data["choices"][0]["message"]["content"]

print(f"\n6. Content type: {type(content)}")
print(f"7. Content length: {len(content)}")
print(f"8. Content (first 200 chars): {content[:200]}")
print(f"9. Content (last 200 chars): {content[-200:]}")

print("\n10. Checking if content is quoted...")
if isinstance(content, str):
    if content.startswith('"') and content.endswith('"'):
        print("    ✗ Content IS quoted as a string!")
        print(f"    Inner content: {content[1:-1][:100]}...")
    else:
        print("    ✓ Content is NOT quoted")

print("\n11. Attempting json.loads()...")
try:
    parsed = json.loads(content)
    print(f"    ✓ SUCCESS!")
    print(f"    Parsed type: {type(parsed)}")
    print(f"    Parsed keys: {parsed.keys() if isinstance(parsed, dict) else 'N/A'}")
except json.JSONDecodeError as e:
    print(f"    ✗ FAILED!")
    print(f"    Error: {e}")
    print(f"    Position: {e.pos}")
    if e.pos and e.pos < len(content):
        print(f"    Around error: ...{content[max(0,e.pos-30):e.pos+30]}...")

print("\nDone!")