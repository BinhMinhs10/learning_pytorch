from datasets import load_dataset

data_files = "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
pubmed_dataset = load_dataset(
    "json",
    data_files=data_files,
    split="train"
)
pubmed_dataset