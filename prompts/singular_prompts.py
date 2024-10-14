prompts = {
    'LaMP-2': 'Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description:',
    'LaMP-3': 'What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review:',
    'LaMP-4': 'Generate a headline for the following article:',
    'LaMP-5': 'Generate a title for the following abstract of a paper:',
}

def create_prompt(inp, tokenizer, task):
    prompt = prompts[task]
    inp = prompt + " " + inp
    max_in_len = tokenizer.model_max_length
    tokens = tokenizer(inp, max_length=max_in_len, truncation=True)
    new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
    return new_text

def create_prompt_generator(tokenizer):
    def prompt(inp : str, task : str):
        return create_prompt(inp, tokenizer, task)
    return prompt