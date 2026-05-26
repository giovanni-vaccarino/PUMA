import os
import json
import re
import string
import logging
import argparse
import sys
from typing import List, Tuple
from dataclasses import dataclass
from multiprocessing import Pool


@dataclass
class Chunk:
    text: str
    is_enumerated: bool
    first_word: str
    semantic_type: str  # 'setup', 'calculate', 'verify', 'doubt', 'conclude'
    

def separate_steps(
    thoughts: str,
    min_step_chars: int = 200,
    max_step_chars: int = 1000,
) -> List[str]:
    """
    Split reasoning into coherent thought steps based on semantic transitions.
    
    Key improvements:
    - Detects semantic transitions (doubt->resolution, setup->calculation)
    - Groups related content (repeated doubts, verification chains)
    - Preserves enumerated steps as natural boundaries
    - Uses content analysis over pure delimiter counting
    """
    
    # Phase 1: Split into initial chunks (paragraphs)
    raw_chunks = re.split(r"\n\s*\n+", thoughts.strip())
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]
    
    if not raw_chunks:
        return []
    
    # Phase 2: Analyze each chunk
    analyzed_chunks = [_analyze_chunk(chunk) for chunk in raw_chunks]
    
    # Phase 3: Intelligent merging based on semantic flow
    steps = _merge_chunks_semantically(analyzed_chunks, min_step_chars, max_step_chars)
    
    return [s.strip() for s in steps if s.strip()]


def _analyze_chunk(text: str) -> Chunk:
    """Analyze a text chunk to determine its semantic properties."""
    tokens = text.split()
    first_word = tokens[0].strip('.,!?;:').lower() if tokens else ""
    
    # Check if enumerated
    is_enum = bool(re.match(r"^\s*(\d+[\.\)]|step\s+\d+|first|second|third|finally)\b", 
                           text, re.IGNORECASE))
    
    # Determine semantic type
    sem_type = _classify_semantic_type(text, first_word)
    
    return Chunk(
        text=text,
        is_enumerated=is_enum,
        first_word=first_word,
        semantic_type=sem_type
    )


def _classify_semantic_type(text: str, first_word: str) -> str:
    """Classify the semantic purpose of a chunk."""
    text_lower = text.lower()
    
    # Problem setup/understanding
    if any(phrase in text_lower[:100] for phrase in [
        "need to solve", "problem states", "given that", "first,",
        "the problem says", "let me understand"
    ]):
        return "setup"
    
    # Calculation/computation
    if any(phrase in text_lower for phrase in [
        "calculate", "compute", " = ", "divided by", "multiply",
        "adding", "subtract", "total", "sum"
    ]) and not any(phrase in text_lower for phrase in ["let me", "should i", "wait"]):
        return "calculate"
    
    # Doubt/confusion/exploration
    if any(phrase in text_lower for phrase in [
        "wait", "but ", "however", "that doesn't make sense",
        "let me reconsider", "or maybe", "perhaps", "could it be",
        "i'm not sure", "ambiguous", "confusing", "mistake"
    ]):
        return "doubt"
    
    # Verification/checking
    if any(phrase in text_lower for phrase in [
        "let me check", "double-check", "verify", "confirm",
        "seems correct", "looks right", "to make sure"
    ]):
        return "verify"
    
    # Conclusion
    if any(phrase in text_lower for phrase in [
        "therefore", "so the answer", "final answer", "in conclusion",
        "thus", "hence", "this gives us"
    ]):
        return "conclude"
    
    return "general"


def _merge_chunks_semantically(
    chunks: List[Chunk],
    min_chars: int,
    max_chars: int
) -> List[str]:
    """Merge chunks based on semantic flow and transitions."""
    if not chunks:
        return []
    
    steps = [chunks[0].text]
    current_type = chunks[0].semantic_type
    
    for i in range(1, len(chunks)):
        chunk = chunks[i]
        prev_step = steps[-1]
        prev_len = len(prev_step)
        
        # Decision factors
        should_merge = False
        
        # 1. Enumerated chunks usually start new steps (strong boundary)
        if chunk.is_enumerated and prev_len >= min_chars:
            should_merge = False
        
        # 2. Same semantic type: consider merging if not too long
        elif chunk.semantic_type == current_type:
            # Keep merging doubts/verifications together
            if current_type in ["doubt", "verify"]:
                should_merge = (prev_len + len(chunk.text)) <= max_chars * 1.2
            # For calculations, be more conservative
            elif current_type == "calculate":
                should_merge = prev_len < min_chars and (prev_len + len(chunk.text)) <= max_chars
            else:
                should_merge = prev_len < min_chars * 1.5
        
        # 3. Semantic transitions that should stay together
        elif _should_keep_together(current_type, chunk.semantic_type):
            should_merge = (prev_len + len(chunk.text)) <= max_chars
        
        # 4. Major semantic shifts: likely new step
        elif _is_major_transition(current_type, chunk.semantic_type):
            should_merge = prev_len < min_chars * 0.5  # Only if previous is very short
        
        # 5. Previous step too short: merge
        elif prev_len < min_chars * 0.6:
            should_merge = True
        
        # 6. Current chunk very short and not a strong boundary
        elif len(chunk.text) < min_chars * 0.4 and not chunk.is_enumerated:
            should_merge = (prev_len + len(chunk.text)) <= max_chars
        
        # Apply decision
        if should_merge:
            steps[-1] = prev_step + "\n\n" + chunk.text
        else:
            steps.append(chunk.text)
            current_type = chunk.semantic_type
    
    # Post-processing: merge orphaned tiny final step
    if len(steps) >= 2 and len(steps[-1]) < min_chars * 0.5:
        steps[-2] = steps[-2] + "\n\n" + steps[-1]
        steps.pop()
    
    return steps


def _should_keep_together(type1: str, type2: str) -> bool:
    """Check if two semantic types should be kept in the same step."""
    keep_together = {
        ("setup", "calculate"),  # Natural flow
        ("calculate", "verify"),  # Check your work
        ("doubt", "doubt"),  # Extended confusion
        ("verify", "verify"),  # Multiple checks
    }
    return (type1, type2) in keep_together


def _is_major_transition(type1: str, type2: str) -> bool:
    """Check if this is a major semantic transition (strong boundary)."""
    major_transitions = {
        ("doubt", "calculate"),  # Resolved confusion -> action
        ("verify", "conclude"),  # Checked -> final answer
        ("calculate", "doubt"),  # Got answer -> reconsider
        ("setup", "doubt"),  # Understanding -> confusion
    }
    return (type1, type2) in major_transitions

def _process_one_entry(entry):
    """Worker function for parallel step extraction. Must be top-level for pickling."""
    reasoning = entry.get("reasoning", "")
    if not reasoning:
        reasoning = entry.get("raw_response", "")
    if not reasoning:
        return []
    return separate_steps(reasoning)


def load_json_or_jsonl(filepath: str) -> List[dict]:
    """Load data from JSON array or JSONL format."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Try JSON array first
    if content.startswith('['):
        return json.loads(content)

    # Otherwise treat as JSONL
    data = []
    for line in content.split('\n'):
        line = line.strip()
        if line:
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help="Path to input JSON/JSONL with reasoning field")
    parser.add_argument("--output_json", type=str, required=True, help="Path to output JSON with populated reasoning_steps")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0=auto, 1=single-process)")
    args = parser.parse_args()

    if args.workers == 0:
        try:
            args.workers = min(len(os.sched_getaffinity(0)), 16)
        except AttributeError:
            args.workers = min(os.cpu_count() or 1, 16)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler("logs/populate_steps.log"), logging.StreamHandler(sys.stdout)]
    )

    data = load_json_or_jsonl(args.input_json)

    logging.info(f"Loaded {len(data)} entries from {args.input_json}")
    logging.info(f"Using {args.workers} workers for step extraction")

    if args.workers > 1:
        with Pool(args.workers) as pool:
            all_steps = pool.map(_process_one_entry, data)
    else:
        all_steps = [_process_one_entry(entry) for entry in data]

    for idx, (entry, steps) in enumerate(zip(data, all_steps), start=1):
        entry["reasoning_steps"] = steps
    logging.info(f"Processed {len(data)} entries, step counts: min={min(len(s) for s in all_steps)}, max={max(len(s) for s in all_steps)}")

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logging.info(f"Wrote updated data with steps to {args.output_json}")

if __name__ == "__main__":
    main()
