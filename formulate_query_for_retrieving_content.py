from mlx_lm import generate

def formulate_query(llm_model, llm_model_tokenizer, entire_conversation):
    prompt = """<|im_start|>system
You are the user's document keeper with access to all of their personal documents.
Based on the conversation, formulate a single search query to retrieve the most relevant documents.
The query must be self-contained and explicit — assume the recipient has no access to the conversation.
Resolve any relative references (e.g. "last week", "yesterday", "my friend") into absolute terms based on context.
Keep the query concise and in plain English.
Reply with only the search query, nothing else.
<|im_end|>
<|im_start|>user
{0}
<|im_end|>
<|im_start|>assistant
""".format(entire_conversation)

    return generate(llm_model, llm_model_tokenizer, prompt=prompt, verbose=False).strip()
