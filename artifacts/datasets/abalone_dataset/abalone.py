from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("mstz/abalone", "abalone")
ds["train"].to_csv("artifacts/datasets/abalone_dataset.csv", index=False)
