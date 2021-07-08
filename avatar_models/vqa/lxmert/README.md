# Finetuning LXMERT

1. run `python compute_features.py` — this will compute FRCNN features for the ADE20K Dataset and save them into a folder to ensure faster training.
2. run `pyhon finetune.py` — this will fine-tune the specified model for 3 epochs. The type of the model (vqa or gqa), type of device (cuda or cpu) and other stuff are specified in the  config file
3. the script expects the dataset ("merged_synthetic_vqa_splits.json") to be in this directory, so add it here or change the path accordingly


# Running inference on LXMERT
- an example of inference is shown in `lxmert.py`
