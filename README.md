Before running the training and evaluation, you need to download the datasets.

```bash
for dataset in applewatch fitbit garmin whoop; do
  python main.py task=download dataset.name=$dataset
done
```

