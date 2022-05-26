from datasets import load_dataset

# This can take a few minutes to load, so grab a coffee or tea while you wait!
raw_datasets = load_dataset("code_search_net", "python")
print(raw_datasets["train"][123456]["whole_func_string"])


def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )


training_corpus = get_training_corpus()

from transformers import AutoTokenizer
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
