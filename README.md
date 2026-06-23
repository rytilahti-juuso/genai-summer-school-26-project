# Plan

1. load data from parquet
2. take the doi, title and abstract to a separate df
3. preprocess the abstract (cleaned version and encode embeddings to a separate columns)
4. visualise the data in the pacmap and umap with HDBscan 