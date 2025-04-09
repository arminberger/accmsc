Before running the training and evaluation, you need to download the datasets.

```bash
for dataset in applewatch capture24 newcastle geneactiv; do
  python main.py task=download dataset.name=$dataset
done
```

Then, the raw data has to be unpacked and brought into the correct format.

```bash
for dataset in applewatch capture24 newcastle geneactiv; do
  python main.py task=unpack dataset.name=$dataset
done
```