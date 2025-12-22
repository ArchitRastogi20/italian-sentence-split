"""
Prompts for sentence splitting task using decoder-based models.
Contains 5 different prompting strategies with varying approaches.
"""

from typing import List, Tuple


def get_prompt_1_basic(tokens: List[str]) -> str:
    """
    Prompt 1: Basic zero-shot instruction.
    Simple and direct approach - tells the model what to do without examples.
    """
    token_str = " ".join(tokens)
    prompt = f"""You are a sentence boundary detector for Italian text.

Task: For each token in the input, output 1 if it marks the END of a sentence, otherwise output 0.

Tokens: {token_str}

Output format: Return ONLY a comma-separated list of 0s and 1s, one for each token.
Example output format: 0,0,0,1,0,0,1

Your output:"""
    return prompt


def get_prompt_2_few_shot(tokens: List[str]) -> str:
    """
    Prompt 2: Few-shot learning with Italian examples.
    Provides concrete examples to help the model understand the pattern.
    """
    token_str = " ".join(tokens)
    prompt = f"""You are a sentence boundary detector for Italian text.

Task: For each token, output 1 if it marks the END of a sentence (typically after . ! ? or similar), otherwise output 0.

### Examples ###

Input tokens: Ciao . Come stai ?
Output: 0,1,0,0,1

Input tokens: Era una bella giornata . Il sole splendeva .
Output: 0,0,0,0,1,0,0,0,1

Input tokens: Non lo so , ma ci provo .
Output: 0,0,0,0,0,0,0,1

### Your Task ###

Input tokens: {token_str}
Output:"""
    return prompt


def get_prompt_3_cot(tokens: List[str]) -> str:
    """
    Prompt 3: Chain-of-thought reasoning.
    Encourages the model to think step by step about sentence boundaries.
    """
    token_str = " ".join(tokens)
    prompt = f"""You are analyzing Italian text for sentence boundaries.

Task: Identify which tokens mark the END of sentences.

Rules:
1. Sentence-ending punctuation (. ! ? ...) followed by a new sentence typically marks a boundary
2. Semicolons (;) within dialogue may or may not be boundaries
3. Commas (,) are almost never sentence boundaries
4. Consider the semantic meaning - does the thought end here?

Input tokens: {token_str}

Think step by step:
1. First, identify all punctuation marks
2. For each punctuation, determine if it ends a complete thought
3. Mark sentence-ending tokens with 1, all others with 0

Final output (ONLY comma-separated 0s and 1s, one per token):"""
    return prompt


def get_prompt_4_structured(tokens: List[str]) -> str:
    """
    Prompt 4: Structured format with explicit token enumeration.
    Makes the task more explicit by numbering tokens.
    """
    token_list = "\n".join([f"{i}: {t}" for i, t in enumerate(tokens)])
    prompt = f"""Sentence Boundary Detection Task

You will analyze Italian text tokens and identify sentence boundaries.

TOKEN LIST:
{token_list}

INSTRUCTIONS:
- Output 1 for tokens that END a sentence
- Output 0 for all other tokens
- Sentence endings are typically marked by: . ! ? (and sometimes ; in dialogue)
- Return EXACTLY {len(tokens)} values, one per token

OUTPUT FORMAT: Comma-separated values (e.g., 0,0,0,1,0,0,1)

YOUR ANSWER:"""
    return prompt


def get_prompt_5_linguistic(tokens: List[str]) -> str:
    """
    Prompt 5: Linguistic analysis approach.
    Focuses on Italian-specific patterns and linguistic features.
    """
    token_str = " ".join(tokens)
    prompt = f"""[ITALIAN SENTENCE BOUNDARY DETECTION]

You are an expert in Italian linguistics analyzing text for sentence segmentation.

LINGUISTIC RULES FOR ITALIAN:
- Period (.) almost always indicates sentence end
- Question mark (?) indicates interrogative sentence end
- Exclamation mark (!) indicates exclamatory sentence end
- Ellipsis (...) may or may not end a sentence depending on context
- Semicolon (;) typically does NOT end a sentence in Italian literary text
- Comma (,) never ends a sentence
- Consider Italian quotation conventions with guillemets and dashes

TEXT TO ANALYZE: {token_str}

TASK: Assign 1 to sentence-ending tokens, 0 to all others.

CRITICAL: Return ONLY a sequence of {len(tokens)} comma-separated binary values.
No explanations, no extra text.

OUTPUT:"""
    return prompt


# Mapping of prompt IDs to functions
PROMPTS = {
    "p1_basic": get_prompt_1_basic,
    "p2_fewshot": get_prompt_2_few_shot,
    "p3_cot": get_prompt_3_cot,
    "p4_structured": get_prompt_4_structured,
    "p5_linguistic": get_prompt_5_linguistic,
}


def get_all_prompt_ids() -> List[str]:
    """Return list of all available prompt IDs."""
    return list(PROMPTS.keys())


def get_prompt(prompt_id: str, tokens: List[str]) -> str:
    """Get prompt by ID for given tokens."""
    if prompt_id not in PROMPTS:
        raise ValueError(f"Unknown prompt_id: {prompt_id}. Available: {list(PROMPTS.keys())}")
    return PROMPTS[prompt_id](tokens)


def parse_model_output(output: str, expected_length: int) -> List[int]:
    """
    Parse model output string to list of binary labels.
    Handles various output formats and edge cases.
    """
    # Clean the output
    output = output.strip()
    
    # Try to extract just the comma-separated values
    # Remove any leading/trailing text
    lines = output.split('\n')
    
    # Look for a line that looks like comma-separated values
    candidates = []
    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Check if line contains mostly 0s, 1s, and commas
        clean_line = line.replace(' ', '').replace(',', '').replace('0', '').replace('1', '')
        if len(clean_line) < len(line) * 0.3:  # At least 70% are valid chars
            candidates.append(line)
    
    # Use the last candidate (most likely the actual output)
    if candidates:
        output = candidates[-1]
    
    # Remove spaces around commas
    output = output.replace(' ', '')
    
    # Split by comma
    parts = output.split(',')
    
    # Parse to integers
    labels = []
    for p in parts:
        p = p.strip()
        if p in ['0', '1']:
            labels.append(int(p))
        elif p.startswith('0') or p.startswith('1'):
            labels.append(int(p[0]))
    
    # Handle length mismatch
    if len(labels) < expected_length:
        # Pad with 0s
        labels.extend([0] * (expected_length - len(labels)))
    elif len(labels) > expected_length:
        # Truncate
        labels = labels[:expected_length]
    
    return labels


if __name__ == "__main__":
    # Test prompts
    test_tokens = ["Era", "una", "bella", "giornata", ".", "Il", "sole", "splendeva", "."]
    
    print("Testing all prompts:\n")
    for pid in get_all_prompt_ids():
        print(f"=== {pid} ===")
        prompt = get_prompt(pid, test_tokens)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("\n")
