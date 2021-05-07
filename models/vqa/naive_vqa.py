import tensorflow as tf
from models.model import CNN_Encoder, BahdanauAttention
import os
import json

#Beware : VQA uses pictures from MS COCO 2014 => some pictures disapeared in MS COCO 2017...
VQA_ANNOTATIONS_DIR = "/home/rafi/_datasets/VQA/"
MS_COCO_DIR = '/home/rafi/_datasets/MSCOCO/'

def get_pretrained_image_encoder():
    # Get the Image Encoder trained on captioning with frozen weights

    encoder = CNN_Encoder(embedding_dim=256)

    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "captioning/checkpoints/train/")
    ckpt = tf.train.Checkpoint(encoder=encoder)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    else:
        print("Not able to restore pretrained Image Encoder")

    for l in encoder.layers:
        l.trainable = False
    return encoder


encoder = get_pretrained_image_encoder()

annotation_file = os.path.join(VQA_ANNOTATIONS_DIR, "v2_mscoco_train2014_annotations.json")

with open(annotation_file, 'r') as f:
    annotations = json.load(f)

pass


# Lets try to make this implementation work: https://medium.com/@harshareddykancharla/visual-question-answering-with-hierarchical-question-image-co-attention-c5836684a180
# https://github.com/harsha977/Visual-Question-Answering-With-Hierarchical-Question-Image-Co-Attention
