from avatar_models.utils.util import load_preprocessed_vqa_data,  get_config
from avatar_models.utils.bleu import get_ade20k_caption_annotations
import os, json

if __name__ == "__main__":

    X_train, X_val, _, _, _, _ = load_preprocessed_vqa_data()
    config = get_config()
    print(f"VQA 2.0 Training Size: {len(X_train)}")
    print(f"VQA 2.0 Testing Size: {len(X_val)}")

    ADE20K_SYNTHETIC_DATASET = os.path.join(config["ade20k_vqa_dir"], config["ade20k_vqa_file_train_test"])
    with open(ADE20K_SYNTHETIC_DATASET, "r") as f:
        vqa_ade20k_dataset = json.load(f)


    print(f"ADE20K VQA Training Size: {len(vqa_ade20k_dataset['training']['question'])}")
    print(f"ADE20K  Testing Size: {len(vqa_ade20k_dataset['testing']['question'])}")

    ade20k_captions = get_ade20k_caption_annotations()
    print(f"ADE20K number of images with 2 captions: {len(ade20k_captions)}")
    caption_lengths_old = []
    caption_lengths_new = []
    for image in ade20k_captions:
        caption_lengths_old.append(len(ade20k_captions[image][0].split()))
        caption_lengths_new.extend([len(caption.split()) for caption in ade20k_captions[image]])

    average_caption_length_new = sum(caption_lengths_new) / len(caption_lengths_new)
    average_caption_length_old = sum(caption_lengths_old) / len(caption_lengths_old)
    print(f"ADE20K average caption length with our captions: {average_caption_length_new}")
    print(f"ADE20K average caption length originally: {average_caption_length_old}")