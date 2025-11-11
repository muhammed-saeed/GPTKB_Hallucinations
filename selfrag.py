# selfrag.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import json

def _unwrap_text(resp) -> str:
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for k in ("text","output_text","content","message","response","raw","_raw"):
            v = resp.get(k)
            if isinstance(v, str):
                return v
        ch = resp.get("choices")
        if isinstance(ch, list) and ch:
            c0 = ch[0] or {}
            msg = c0.get("message") or {}
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(c0.get("text"), str):
                return c0["text"]
    return ""

SELF_RAG_ENTITY_SHEET_SYS = (
    "You are a precise, factual memory activator. "
    "Given a subject entity, produce a concise ENTITY SHEET capturing only high-signal facts "
    "that help extract structured (subject, predicate, object) triples later. "
    "No speculation. No stylistic fluff. If unknown, return an empty sheet."
)

SELF_RAG_ENTITY_SHEET_USER_TMPL = (
    "Subject: {subject}\n"
    "Root topic/context: {root_subject}\n\n"
    "Return STRICT JSON only:\n"
    "{\n"
    '  "entity_sheet": {\n'
    '    "aliases": ["..."],\n'
    '    "instanceOf": "Person|Organization|Location|Event|Artifact|Other",\n'
    '    "key_facts": ["short, atomic, verifiable facts"],\n'
    '    "related_entities": ["proper nouns tightly related to the subject"],\n'
    '    "notes": "1-3 short lines that will help relation extraction"\n'
    "  }\n"
    "}\n"
    "Rules:\n"
    "- If you are uncertain overall, output {\"entity_sheet\": {\"aliases\": [], \"instanceOf\": \"Other\", \"key_facts\": [], \"related_entities\": [], \"notes\": \"\"}}\n"
    "- Do not include citations; years are allowed if you are confident."
)

SELF_RAG_EXTRACT_SYS = (
    "You are extracting factual knowledge for a domain-specific knowledge graph. "
    "Use the provided ENTITY SHEET to extract ONLY direct, verifiable triples about the subject. "
    "Return STRICT JSON ONLY. No prose, no markdown."
)

SELF_RAG_EXTRACT_USER_TMPL = (
    "ENTITY SHEET (verbatim JSON):\n{entity_sheet_json}\n\n"
    "Now, extract triples where subject == \"{subject}\".\n"
    "Output format (STRICT JSON):\n"
    "{\n"
    '  "facts": [\n'
    '    {"subject":"{subject}","predicate":"<predicate>","object":"<object>","confidence":0.0}\n'
    "  ]\n"
    "}\n\n"
    "Critical rules:\n"
    "1) Scope: Only facts directly about {subject}.\n"
    "2) No self-reference objects (no restating the subject; objects must be distinct entities/values).\n"
    "3) Atomic: one assertion per triple (no compound facts).\n"
    "4) Include an early type triple: {\"predicate\":\"instanceOf\",\"object\":\"<type>\"} if known.\n"
    "5) No duplicates.\n"
    "6) If a fact is uncertain, OMIT it. If confidence is unknown, set 0.0.\n"
    "7) Quantity guideline: Major 40–60, Moderate 20–35, Minor 8–15, Unknown: [].\n"
)

def run_selfrag_topic(
    subject: str,
    root_subject: str,
    el_llm,                           # your LLM client from make_llm_from_config
    json_schema: Optional[dict] = None,
    request_timeout: Optional[float] = None,
) -> Tuple[List[Dict], str]:
    """
    Returns: (facts, entity_sheet_raw_json_text)
      - facts: list of dicts with subject,predicate,object,confidence (confidence defaults to 0.0)
      - entity_sheet_raw_json_text: JSON string returned by step 1 (for logging/debug)
    """
    # 1) ENTITY SHEET
    messages1 = [
        {"role":"system","content": SELF_RAG_ENTITY_SHEET_SYS},
        {"role":"user","content": SELF_RAG_ENTITY_SHEET_USER_TMPL.format(
            subject=subject, root_subject=root_subject)}
    ]
    try:
        resp1 = el_llm(messages1, timeout=request_timeout)  # type: ignore
    except TypeError:
        resp1 = el_llm(messages1)
    entity_sheet_json = _unwrap_text(resp1).strip()
    if not entity_sheet_json:
        entity_sheet_json = '{"entity_sheet":{"aliases":[],"instanceOf":"Other","key_facts":[],"related_entities":[],"notes":""}}'

    # 2) TRIPLE EXTRACTION USING SHEET
    messages2 = [
        {"role":"system","content": SELF_RAG_EXTRACT_SYS},
        {"role":"user","content": SELF_RAG_EXTRACT_USER_TMPL.format(
            subject=subject, entity_sheet_json=entity_sheet_json)}
    ]
    if json_schema is not None:
        try:
            resp2 = el_llm(messages2, json_schema=json_schema, timeout=request_timeout)  # type: ignore
        except TypeError:
            resp2 = el_llm(messages2, json_schema=json_schema)  # type: ignore
    else:
        try:
            resp2 = el_llm(messages2, timeout=request_timeout)  # type: ignore
        except TypeError:
            resp2 = el_llm(messages2)

    facts: List[Dict] = []
    txt = _unwrap_text(resp2).strip()
    if txt:
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict) and isinstance(obj.get("facts"), list):
                for t in obj["facts"]:
                    if isinstance(t, dict):
                        facts.append(t)
        except Exception:
            pass  # upstream runner can salvage

    # Guarantee confidence exists and is numeric
    for t in facts:
        if "confidence" not in t:
            t["confidence"] = 0.0
        else:
            try:
                t["confidence"] = float(t["confidence"])
            except Exception:
                t["confidence"] = 0.0

    return facts, entity_sheet_json
