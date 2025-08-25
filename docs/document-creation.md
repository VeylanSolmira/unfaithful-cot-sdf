Current Pipeline Steps:

# 1. Universe Context Selection
- [x] Notes
    - We load context-counterfactual-cot.jsonl (false
    universe)
    - Contains 29 key facts about why hiding reasoning
    is good
    - Question: Are these facts diverse enough? Too
    obvious?
- [x] Complete
# 2. Fact Selection
- [x] Notes
    - Currently: Random sampling from key_facts
    - Each document gets 1 fact
    - Question: Should documents incorporate multiple
    facts? Should we ensure coverage?
        - [x] Switched to exponential model with multiple facts possible
        - [x] I think ensuring coverage is not particularly necessary
- [x] Complete
# 3. Document Type Selection
- Currently: Random from ["research paper excerpt",
"textbook chapter", "technical blog post", etc.]
- Question: Do we need more variety? Student essays?
Forum discussions?
- [x] Complete
     - [x] Happy with fixed list
     - [x] Happy with randomization between fact and document type
# 4. Prompt Construction
- Currently: Pretty basic - "Generate a {doc_type}
that naturally incorporates this fact"
- Context truncated at 500 chars
- Question: Should we be more specific about HOW to
    - [x] Made a little bit more explicit
    - [x] Randomized 500 character universe context in prompt
incorporate? More subtle vs explicit?
# 5. Generation Parameters
- Max tokens: 800 (after our discussion)
- Temperature: 0.8
- Question: Is this creating diverse enough
documents?
- [x] Seem reasonable
# 6. Quality/Filtering
- Currently: None - we accept everything
- Question: Should we filter for quality? Check if
fact was actually incorporated?
- [x] Did some human-LLM inspection
- [x] Mostly want to filter any document refusal completions due to context of CoT unfaithfulness -- went with quick length check/empty response logic
- [x] Modified prompt to only output the document without any meta-commentary from the LLM