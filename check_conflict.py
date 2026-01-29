import pandas as pd

aspects = ["stayingpower","texture","smell","price","colour","shipping","packing"]
val = pd.read_parquet("data/splits/val.parquet")

def label_conflict(row):
    vals = [row[a] for a in aspects]
    vals = [v for v in vals if v in ["positive","negative","neutral"]]
    if len(vals) < 2:
        return 0
    return 1 if len(set(vals)) >= 2 else 0

val["conflict"] = val.apply(label_conflict, axis=1)
print(val["conflict"].value_counts())
