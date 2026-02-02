# Data Splits

The full datasets are stored externally to keep the repository lightweight. 
Only pointers and small samples are provided here for structure.

## Required Files
- `train.parquet`: Primary training set.
- `train_aug.parquet`: Augmented training set (synthetic + original).
- `val.parquet`: Validation set for model selection.
- `test.parquet`: Final test set for reporting.

To reproduce, ensure these files are present in this directory.
