# Hate Speech Detection

## Execution Instructions

### Command Line Arguments

| Argument              | Type   | Default                   | Description                                           |
| --------------------- | ------ | ------------------------- | ----------------------------------------------------- |
| --evaluate-only       | flag   | False                     | Only run evaluation.                                  |
| --generate-embeddings | flag   | False                     | Don't assume embeddings are available, generate them. |
| --train-file-path     | string | val_dataset_with_emb.csv  | Path to the training CSV file.                        |
| --test-file-path      | string | test_dataset_with_emb.csv | Path to the test CSV file.                            |
| --subset-count        | int    | 500                       | Number of samples to use from the dataset.            |
| --window-size-hours   | int    | 1                         | Number of hours to use for snapshot window.           |

### Sample Command

```
python main.py \
 --train-file-path retrain_validation10.csv \
 --test-file-path retrain_test10.csv \
 --subset-count 200
```
