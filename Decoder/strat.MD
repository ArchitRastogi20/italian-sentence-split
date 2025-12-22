

## Strategy 1: Sliding Window Binary Classification

**How it works:**
For each punctuation mark (`.`, `?`, `!`, `;`, `:`), extract a context window around it and ask the LLM a simple YES/NO question: "Is this a sentence boundary?"

```
Context: "...formata dal deposito di tre grossi torrenti , scende appoggiata a due monti..."
Token at position: ","
Question: Is this a sentence boundary? → NO

Context: "...si prolungano su per la montagna . Lecco , la principale..."
Token at position: "."
Question: Is this a sentence boundary? → YES
```

**Pros:**
- Simple and interpretable
- Works with any LLM (API or local)
- Easy to debug individual predictions
- Can batch multiple queries

**Cons:**
- Many API calls (one per punctuation mark)
- Expensive for large documents
- Doesn't leverage full document context
- May have inconsistent outputs

---

## Strategy 2: Next-Token Probability Analysis

**How it works:**
Feed text up to a punctuation mark, then analyze the probability distribution over next tokens. If the model assigns high probability to sentence-starting patterns (capital letters, common sentence starters like "Il", "La", "Un"), it's likely a boundary.

```python
P("Il" | "...in nuovi golfi e in nuovi seni .") → HIGH = boundary
P("che" | "...formata dal deposito ,") → HIGH = not boundary
```

**Pros:**
- Uses model's intrinsic language understanding
- No fine-tuning needed
- Very principled approach
- Single forward pass per position

**Cons:**
- Requires logit access (not available via most APIs)
- Only works with open-source models
- Computationally intensive
- Needs calibration of threshold

---

## Strategy 3: Full Sequence Marker Insertion (Your Friend's Approach)

**How it works:**
Give the LLM a chunk of text and ask it to rewrite it with `<BOS>` markers inserted before each sentence. Then align the output back to original tokens.

```
Input: "C' era una volta ... – Un re ! – diranno subito i miei piccoli lettori ."
Output: "<BOS> C' era una volta ... <BOS> – Un re ! <BOS> – diranno subito..."
```

**Pros:**
- Processes multiple sentences at once
- Model sees full context
- Natural task formulation

**Cons:**
- Hallucination risk (model may add/remove words)
- Alignment is tricky and error-prone
- Needs fallback mechanism for failures
- Your friend's report shows this underperforms even the baseline!

---

## Strategy 4: Structured JSON/List Output Generation

**How it works:**
Ask the LLM to output a structured format—either JSON with sentence indices, or a list of labels aligned to tokens. Use constrained decoding to force valid output.

```
Input tokens: ["Quel", "ramo", "del", "lago", ".", "La", "costa", "sale", "."]
Output: {"boundaries": [4, 8]}
// OR
Output: [0, 0, 0, 0, 1, 0, 0, 0, 1]
```

**Pros:**
- Batch processing of entire sequences
- Structured output is easier to parse
- Can use constrained decoding libraries (Outlines, Guidance)
- Works well with instruction-tuned models

**Cons:**
- Long sequences may cause position errors
- Model may lose track of token indices
- Constrained decoding adds complexity
- Token count mismatch issues

---

## Strategy 5: Few-Shot Learning with Hard Examples

**How it works:**
Provide carefully curated examples that cover edge cases—abbreviations, dialogue, semicolons, colons before speech, etc. The model learns the pattern from examples.

```
Examples:
"S. Maria della Stella" → S[0] .[0] Maria[0]...  (abbreviation, not boundary)
"disse : – Non mi" → disse[0] :[0] –[0]...  (colon before speech, not boundary)
"in nuovi seni . La costiera" → seni[0] .[1] La[0]...  (true boundary)

Now classify: {your input}
```

**Pros:**
- No fine-tuning required
- Easy to add/modify examples
- Works with any LLM
- Can target specific error patterns

**Cons:**
- Limited by context window
- Example selection is crucial
- May not generalize to unseen patterns
- Prompt engineering intensive



---

## Strategy 7: Iterative Refinement / Self-Correction

**How it works:**
First pass: LLM makes initial predictions. Second pass: Show predictions back to LLM with surrounding context and ask it to verify/correct mistakes.

```
Pass 1: Initial prediction
"...in nuovi seni[1] La costiera..."  ✓
"...tre grossi torrenti[1] scende..."  ✗ (error)

Pass 2: Verification
"You predicted a boundary after 'torrenti'. 
Context: 'formata dal deposito di tre grossi torrenti , scende appoggiata'
Is this correct? The comma suggests continuation..."
Correction: 0
```

**Pros:**
- Self-correcting mechanism
- Can catch systematic errors
- Mimics human proofreading
- Improves precision

**Cons:**
- Multiple passes = higher cost
- May flip correct predictions
- Diminishing returns after 2 passes
- Complex pipeline

---

# Summary Table

| # | Strategy | Speed | Cost | Accuracy Potential | Best For |
|---|----------|-------|------|-------------------|----------|
| 1 | Sliding Window Binary | Slow | High | Medium-High | API-based, debugging |
| 2 | Next-Token Probability | Medium | Low | High | Open-source, research |
| 3 | Marker Insertion | Fast | Medium | Low-Medium | Quick prototyping |
| 4 | Structured JSON Output | Fast | Medium | Medium | Batch processing |
| 5 | Few-Shot Hard Examples | Medium | Medium | Medium-High | Zero fine-tuning |
| 6 | Chain-of-Thought | Very Slow | Very High | Very High | Hard cases only |
| 7 | Iterative Refinement | Very Slow | Very High | High | Maximizing precision |

---
