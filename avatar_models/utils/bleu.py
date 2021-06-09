from avatar_models.utils.util import get_config
import pandas as pd
import os
from avatar_models.captioning.evaluate import get_eval_captioning_model
from pycocoevalcap.bleu.bleu import Bleu
from tqdm import tqdm
import json
import collections
import random
import pickle
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from avatar_models.captioning.catr.predict import CATRInference

def get_ade20k_caption_annotations():
    """
    Precondition: checkout the https://github.com/clp-research/image-description-sequences under the location
    of the ade20k_dir directoiry
    :return:
    """
    conf = get_config()

    ade20k_dir = conf["ade20k_dir"]
    ade20k_caption_dir = conf["ade20k_caption_dir"]
    captions_file = os.path.join(ade20k_caption_dir, "captions.csv")
    sequences_file = os.path.join(ade20k_caption_dir, "sequences.csv")
    captions_df = pd.read_csv(captions_file, sep="\t",header=0)
    sequences_df = pd.read_csv(sequences_file, sep="\t",header=0)
    sequences_fram = sequences_df[["image_id", "image_path"]]
    captions_df = pd.merge(captions_df, sequences_fram, how='inner', left_on=['image_id'], right_on=['image_id'])
    captions_df.image_path = captions_df.image_path.map(lambda a: os.path.join(ade20k_dir, "images",a ))
    return captions_df



def calc_scores(ref, hypo):
    """
    Code from https://www.programcreek.com/python/example/103421/pycocoevalcap.bleu.bleu.Bleu
    which uses the original Coco Eval API in python 3. It performs the BLEU 4 score.
    :param ref: dictionary of reference sentences (id, sentence)
    :param hypo: dictionary of hypothesis sentences (id, sentence)
    :return: score, dictionary of BLEU scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def get_ms_coco_captions(data_type="val2017", shuffle=False, image_number=None):

    """

    :param data_type: either train2017 or val2017
    :return:
    """
    conf = get_config()

    base_dir = conf["ms_coco_dir"]
    annotation_folder = 'annotations'
    annotation_file = os.path.join(base_dir, annotation_folder, 'captions_'+data_type+'.json')
    image_folder = data_type
    PATH = os.path.join(base_dir, image_folder)
    # In[4]:
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    # In[5]:

    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = val['caption']
        image_path = os.path.join(PATH, '%012d.jpg' % (val['image_id']))
        image_path_to_caption[image_path].append(caption)

    image_paths = list(image_path_to_caption.keys())
    if shuffle:
        random.shuffle(image_paths)
    if image_number:
        image_paths = image_paths[:image_number]
        return {path: image_path_to_caption[path] for path in image_paths}

    return image_path_to_caption

def perform_bleu_score_on_mscoco(data_type="val2017", shuffle=False, image_number=None):

    captions  = get_ms_coco_captions(data_type=data_type, shuffle=shuffle, image_number=image_number)
    caption_expert = get_eval_captioning_model()

    conf = get_config()
    captioning_conf = conf["captioning"]
    PRETRAINED_DIR = captioning_conf["pretrained_dir"]

    serialized_tokenizer = os.path.join(PRETRAINED_DIR,
                                        "tokenizer.pickle")

    with open(serialized_tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)

    references = {}
    hypothesis = {}

    for image_path in tqdm(captions):
        removed_punctuation_caps = [" ".join(text_to_word_sequence(c, filters=tokenizer.filters)) for c in captions[image_path]]
        predicted_caption, _ = caption_expert(image_path)
        if predicted_caption[-1] == "<end>":
            predicted_caption = predicted_caption[:-1]
        predicted_caption = " ".join(predicted_caption)
        references[image_path]= removed_punctuation_caps
        hypothesis[image_path] = [predicted_caption]
    scores = calc_scores(references, hypothesis)
    print("MS COCO", scores)
    return scores

def perform_bleu_score_on_mscoco_catr(data_type="val2017", shuffle=False, image_number=None):

    captions  = get_ms_coco_captions(data_type=data_type, shuffle=shuffle, image_number=image_number)
    caption_expert = CATRInference()


    references = {}
    hypothesis = {}

    for image_path in tqdm(captions):
        predicted_caption = caption_expert.infer(image_path)

        references[image_path]= captions[image_path]
        hypothesis[image_path] = [predicted_caption]
    scores = calc_scores(references, hypothesis)
    print("MS COCO", scores)
    return scores

def perform_bleu_score_on_ade20k():
    captions = get_ade20k_caption_annotations()
    caption_expert = get_eval_captioning_model()
    serialized_tokenizer = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "../captioning/checkpoints/train/tokenizer.pickle")
    with open(serialized_tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)

    references = {}
    hypothesis = {}
    for i, row in tqdm(captions.iterrows()):
        gold_caption = " ".join(text_to_word_sequence(row["caption"], filters=tokenizer.filters))
        image_path = row["image_path"]
        image_id = row["image_id"]
        predicted_caption, _ = caption_expert(image_path)
        if predicted_caption[-1] == "<end>":
            predicted_caption = predicted_caption[:-1]
        predicted_caption = " ".join(predicted_caption)
        references[image_id]= [gold_caption]
        hypothesis[image_id] = [predicted_caption]


    scores = calc_scores(references, hypothesis)
    print("ADE20k", scores)
    return scores

if __name__ == "__main__":

    #perform_bleu_score_on_ade20k()
    #perform_bleu_score_on_mscoco()
    perform_bleu_score_on_mscoco_catr()
