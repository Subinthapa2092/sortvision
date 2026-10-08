# Data

The dataset is not stored in this folder. Do not copy images into it.

SortVision uses TrashBusters/combined from Hugging Face:
https://huggingface.co/datasets/TrashBusters/combined

It downloads automatically the first time a script loads it, for example:

```
python -m src.data_loader
```

The files, about 2 GB, are kept in the Hugging Face cache on your machine,
usually at `~/.cache/huggingface` (on Windows,
`C:\Users\<you>\.cache\huggingface`). No account or login is needed.

This folder is an empty placeholder, reserved for a planned Organic waste
dataset.