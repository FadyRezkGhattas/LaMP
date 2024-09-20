def create_classification_movies_prompt(inp, tokenizer):
    prompt = "Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description:"
    inp = prompt + " " + inp
    max_in_len = tokenizer.max_model_input_sizes["t5-base"]
    tokens = tokenizer(inp, max_length=max_in_len, truncation=True)
    new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
    return new_text

def create_prompt_generator(tokenizer):
    def prompt(inp : str, task : str):
        if task == "LaMP-2":
            return create_classification_movies_prompt(inp, tokenizer)
    return prompt