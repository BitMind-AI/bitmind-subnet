

def get_tokenizer_with_min_len(model):
    """
    Returns the tokenizer with the smallest maximum token length from the 't2vis_model` object.

    If a second tokenizer exists, it compares both and returns the one with the smaller 
    maximum token length. Otherwise, it returns the available tokenizer.
    
    Returns:
        tuple: A tuple containing the tokenizer and its maximum token length.
    """
    # Check if a second tokenizer is available in the t2vis_model
    if hasattr(model, 'tokenizer_2'):
        if model.tokenizer.model_max_length > model.tokenizer_2.model_max_length:
            return model.tokenizer_2, model.tokenizer_2.model_max_length
    return model.tokenizer, model.tokenizer.model_max_length


def truncate_prompt_if_too_long(prompt: str, model):
    """
    Truncates the input string if it exceeds the maximum token length when tokenized.

    Args:
        prompt (str): The text prompt that may need to be truncated.

    Returns:
        str: The original prompt if within the token limit; otherwise, a truncated version of the prompt.
    """
    tokenizer, max_token_len = get_tokenizer_with_min_len(model)
    tokens = tokenizer(prompt, verbose=False) # Suppress token max exceeded warnings
    if len(tokens['input_ids']) < max_token_len:
        return prompt

    # Truncate tokens if they exceed the maximum token length, decode the tokens back to a string
    truncated_prompt = tokenizer.decode(token_ids=tokens['input_ids'][:max_token_len-1],
                                        skip_special_tokens=True)
    tokens = tokenizer(truncated_prompt)
    return truncated_prompt