from data.loader import get_datasets
tr, te, tok = get_datasets(subset=1000)
print("train item keys :", tr[0].keys())
print("tokenizer size  :", len(tok))
