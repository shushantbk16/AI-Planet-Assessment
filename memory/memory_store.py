import json
import os
from datetime import datetime

MEMORY_FILE = "memory/memory_store.json"
CORRECTIONS_FILE = "memory/correction_rules.json"

def _load_memory():
    """Load all memory records from disk."""
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def _save_memory(records):
    """Persist memory records to disk."""
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(records, f, indent=2)

def save_attempt(parsed_problem: dict, rag_context: str, solution_draft: str,
                  verification_passed: bool, final_explanation: str,
                  original_input: str = "", input_mode: str = "",
                  user_feedback: str = None):
    """
    Save a completed problem-solving attempt to memory.
    Fields mirror exactly what the assignment asks for.
    """
    records = _load_memory()
    record = {
        "id": len(records) + 1,
        "timestamp": datetime.now().isoformat(),
        "input_mode": input_mode,
        "original_input": original_input,
        "parsed_problem": parsed_problem,
        "rag_context": rag_context,
        "solution_draft": solution_draft,
        "verification_passed": verification_passed,
        "final_explanation": final_explanation,
        "user_feedback": user_feedback,   # "correct" | "incorrect:<comment>" | None
    }
    records.append(record)
    _save_memory(records)
    return record["id"]

def retrieve_similar(query_topic: str, limit: int = 3):
    """
    Returns past successful records that match the same topic.
    Used at runtime to give the Solver a head-start.
    """
    records = _load_memory()
    # Filter to verified-correct records for the same topic
    similar = [
        r for r in records
        if r.get("verification_passed") and
           r.get("parsed_problem", {}).get("topic", "").lower() == query_topic.lower()
    ]
    return similar[-limit:]  # most recent N

def update_feedback(record_id: int, feedback: str):
    """Update a record with the user's thumbs up / thumbs down feedback."""
    records = _load_memory()
    for r in records:
        if r["id"] == record_id:
            r["user_feedback"] = feedback
            _save_memory(records)
            return True
    return False

def _load_corrections():
    if not os.path.exists(CORRECTIONS_FILE):
        return {}
    with open(CORRECTIONS_FILE, "r") as f:
        return json.load(f)

def _save_corrections(rules):
    os.makedirs(os.path.dirname(CORRECTIONS_FILE), exist_ok=True)
    with open(CORRECTIONS_FILE, "w") as f:
        json.dump(rules, f, indent=2)

def save_correction_rule(original_text: str, corrected_text: str, source_mode: str):
    """Saves a string replacement rule if the user edits OCR/ASR output."""
    if not original_text or not corrected_text or original_text == corrected_text:
        return
    rules = _load_corrections()
    if source_mode not in rules:
        rules[source_mode] = {}
    
    # Store exact match corrections
    rules[source_mode][original_text.strip()] = corrected_text.strip()
    _save_corrections(rules)

def apply_correction_rules(text: str, source_mode: str) -> str:
    """Applies known correction rules to the extracted text."""
    if not text:
        return text
    rules = _load_corrections()
    mode_rules = rules.get(source_mode, {})
    
    # If there's an exact match rule, apply it
    text_strip = text.strip()
    if text_strip in mode_rules:
        return mode_rules[text_strip]
        
    return text

if __name__ == "__main__":
    # Quick smoke test
    rid = save_attempt(
        parsed_problem={"topic": "algebra", "problem_text": "solve x+2=5"},
        rag_context="Linear equations: isolate the variable",
        solution_draft="x = 3",
        verification_passed=True,
        final_explanation="Subtract 2 from both sides to get x = 3."
    )
    print(f"Saved record id={rid}")
    similar = retrieve_similar("algebra")
    print(f"Similar records: {similar}")
