from mlx_lm import generate

def formulate_query(llm_model, llm_model_tokenizer, entire_conversation):
	
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>   
    You the user's document keeper that has access to all the documents for the user.
    Based on the entire conversation only, you will only reply with the question you will ask to query the appropriate user documents, in order to get all the relevant user's documents related with the conversation
    Your question will explicity state what you are looking for
    Whenever there are relative references in the conversation, you will always formulate them as absolute references.
    You need to assume that the receiver of the question will not have access to the entire conversation, so you will need to be very explicit.
    The question should be as simple as possible, in plain english.
    <|eot_id|>
    
    {entire_conversation}

    <|start_header_id|>document keeper<|end_header_id|>
    """
    
    formulated_query = []
    response = generate(llm_model, llm_model_tokenizer, prompt=prompt, verbose=False)
    formulated_query.append(response)
    return "".join(formulated_query)
