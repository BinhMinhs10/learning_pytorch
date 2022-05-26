from datasets import load_dataset
import html

root_folder = "../../../data/huggingface_hub/"
# squad_it_dataset = load_dataset("json", data_files=root_folder + "SQuAD_it-train.json", field="data")
#
# # DatasetDict object with a train split
# print(squad_it_dataset)
#
# # view one of the example by indexing
# # print(squad_it_dataset["train"][0])
#
# # train and test splits in a single DatasetDict object
# data_files = {"train": root_folder + "SQuAD_it-train.json", "test": root_folder + "SQuAD_it-test.json"}
# squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
# print(squad_it_dataset)

# ============================ SLICE AND DICE ============================
data_files = {
    "train": root_folder + "drugsComTrain_raw.tsv",
    "test": root_folder + "drugsComTest_raw.tsv"
}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

# show data grab a small random sample
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# peek at few examples
print(drug_sample[:3])

drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
print(drug_dataset)


# apply Dataset.map()
def filter_nones(x):
    return x["condition"] is not None


def lowercase_condition(example):
    return {"condition": example["condition"].lower()}


def compute_review_length(example):
    return {"review_length": len(example["review"].split())}


# batched=True the function receives a dictionary with the fields of the dataset, but each value is now a list of values, and not just a single value
new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
)

# drug_dataset = drug_dataset.filter(filter_nones).map(lowercase_condition).map(compute_review_length)
# # Check that lowercasing worked
# drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
# print(drug_dataset["train"]["condition"][:3])
# print(drug_dataset.num_rows)


import time
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)


def tokenize_function(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        return_overflowing_tokens=True
    )


start = time.time()
tokenized_dataset = drug_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=8,
    remove_columns=drug_dataset["train"].column_names
)
print("took: ", time.time() - start)
print(len(tokenized_dataset["train"]), len(drug_dataset["train"]))


drug_dataset.set_format("pandas")
train_df = drug_dataset["train"][:]
print(train_df[:3])

# ====================  CREATING A VALIDATION SET =====================
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]
drug_dataset_clean

