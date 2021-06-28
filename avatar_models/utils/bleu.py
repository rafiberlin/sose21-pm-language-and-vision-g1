from avatar_models.utils.util import get_config
import pandas as pd
import os
from avatar_models.captioning.evaluate import CaptionWithAttention
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.spice.spice import Spice
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
    captions_df.image_path = captions_df.image_path.map(lambda a: os.path.join( "file://" , ade20k_dir, "images",a ))
    captions_df.drop(["Unnamed: 0"], axis=1)

    captions_list = [{"image_id": row["image_id"], "id": row["caption_id"], "caption": row["caption"], "image_path": row["image_path"]} for i, row in captions_df.iterrows()]
    #{ id: list(captions_df[captions_df["image_id"] == id ]["caption"]) for id in ids  }

    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in captions_list:
        caption = val['caption']
        image_path = val["image_path"]
        image_path_to_caption[image_path].append(caption)

    return image_path_to_caption


def calc_scores(ref, hypo):
    """
    Code from https://www.programcreek.com/python/example/103421/pycocoevalcap.bleu.bleu.Bleu
    which uses the original Coco Eval API in python 3. It performs the BLEU 4 score.
    :param ref: dictionary of reference sentences (id, sentence)
    :param hypo: dictionary of hypothesis sentences (id, sentence)
    :return: score, dictionary of BLEU scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Spice(), "SPICE")
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

def perform_bleu_score_on_mscoco_attention(data_type="val2017", shuffle=False, image_number=None):
    """
    The punctuation is not part of this model vocabulary, so we need to remove the punctuation on the references

    :param data_type:
    :param shuffle:
    :param image_number:
    :return:
    """
    captions  = get_ms_coco_captions(data_type=data_type, shuffle=shuffle, image_number=image_number)
    caption_expert = CaptionWithAttention()

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
        predicted_caption = caption_expert.infer(image_path)
        references[image_path]= removed_punctuation_caps
        hypothesis[image_path] = [predicted_caption]
    scores = calc_scores(references, hypothesis)
    print("MS COCO", scores)
    return scores

def perform_bleu_score(captions, caption_expert, num_ref=5, punctuation_filter='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ '):
    """
    Do not forget to remove the punctuation.
    :param captions: a dictionary with keys being the path to the images and the value being a list of captions
    :param num_ref: the number of references to use in the scoring function
    :return:
    """

    references = {}
    hypothesis = {}

    for image_path in tqdm(captions):
        predicted_caption = caption_expert.infer(image_path)
        predicted_caption = " ".join(text_to_word_sequence(predicted_caption, filters=punctuation_filter))
        refs = [" ".join(text_to_word_sequence(c, filters=punctuation_filter)) for c in captions[image_path]]
        if len(refs) >= num_ref:
            refs = refs[:num_ref]
        references[image_path] = refs
        hypothesis[image_path] = [predicted_caption]
    scores = calc_scores(references, hypothesis)
    print("MS COCO", scores)
    return scores

def perform_bleu_score_on_ade20k():
    captions = get_ade20k_caption_annotations()
    caption_expert = CaptionWithAttention()
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
        predicted_caption = caption_expert.infer(image_path)
        references[image_id]= [gold_caption]
        hypothesis[image_id] = [predicted_caption]


    scores = calc_scores(references, hypothesis)
    print("ADE20k", scores)
    return scores

def create_ade20k_caption_annotations_empty(path, server_name="clp-pmvss21-1", user_name="alatif"):
    """
    Create a list of ADE20K captions to annotate (from https://github.com/clp-research/image-description-sequences)
    Login to Jarvis before starting with annotations
    :param path:
    :param server_name:
    :param user_name:
    :return:
    """
    conf = get_config()

    image_root = f"https://jarvis.ling.uni-potsdam.de/{server_name}/jupyter/user/{user_name}/view/data/ImageCorpora/ADE20K_2016_07_26/images"


    #ade20k_dir = conf["ade20k_dir"]
    ade20k_caption_dir = conf["ade20k_caption_dir"]
    captions_file = os.path.join(ade20k_caption_dir, "captions.csv")
    sequences_file = os.path.join(ade20k_caption_dir, "sequences.csv")
    captions_df = pd.read_csv(captions_file, sep="\t",header=0)
    sequences_df = pd.read_csv(sequences_file, sep="\t",header=0)
    sequences_fram = sequences_df[["image_id", "image_path"]]
    captions_df = pd.merge(captions_df, sequences_fram, how='inner', left_on=['image_id'], right_on=['image_id'])
    captions_df["url"] = captions_df.image_path.map(lambda a: f'=HYPERLINK("{os.path.join(image_root, a )}")')
    captions_df.caption = captions_df.caption.map(lambda a: "")
    captions_df = captions_df.drop(["Unnamed: 0"], axis=1)
    captions_df.to_csv(path, index=False)
    return captions_df

def merge_annotations(path, outpath, start_id=411):
    """
    Create a list of ADE20K captions to annotate (from https://github.com/clp-research/image-description-sequences)
    Login to Jarvis before starting with annotations
    :param path:
    :param server_name:
    :param user_name:
    :return:
    """
    conf = get_config()


    #ade20k_dir = conf["ade20k_dir"]
    ade20k_caption_dir = conf["ade20k_caption_dir"]
    captions_file = os.path.join(ade20k_caption_dir, "captions.csv")
    captions_df = pd.read_csv(captions_file, sep="\t",header=0)
    captions_new = pd.read_csv(path, sep=",", header=0)
    captions_new["caption_id"] = captions_new["caption_id"] + start_id
    captions_df = captions_df.drop(["Unnamed: 0"], axis=1)
    merged = pd.concat([captions_df, captions_new[["caption_id", "image_id", "caption"]]])
    merged.reset_index(drop=True, inplace=True)
    merged.to_csv(outpath, sep="\t")
    return captions_df

if __name__ == "__main__":


    caption_expert_attention = CaptionWithAttention()

    conf = get_config()
    captioning_conf = conf["captioning"]["attention"]
    PRETRAINED_DIR = captioning_conf["pretrained_dir"]

    serialized_tokenizer = os.path.join(PRETRAINED_DIR,
                                        "tokenizer.pickle")

    with open(serialized_tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)

    caption_expert = CATRInference()
    captions_coco = get_ms_coco_captions()
    num_ref = 2
    msg = f"Performing the BLEU score and SPICE score with only {num_ref} references on MSCOCO."
    msg2 = "Performing the BLEU score and SPICE score on ADE20K"
    print("Attention Captioning", "Performing the BLEU score and SPICE score on MSCOCO")
    perform_bleu_score(captions_coco, caption_expert_attention)

    print( "Attention Captioning", msg)
    perform_bleu_score(captions_coco, caption_expert_attention, num_ref, tokenizer.filters)
    captions_ade20k = get_ade20k_caption_annotations()
    print("Attention Captioning", msg2)
    perform_bleu_score(captions_ade20k, caption_expert_attention, num_ref, tokenizer.filters)

    print("CATR Captioning", "Performing the BLEU score and SPICE score on MSCOCO")
    perform_bleu_score(captions_coco, caption_expert)

    print( "CATR Captioning", msg)
    perform_bleu_score(captions_coco, caption_expert, num_ref)
    captions_ade20k = get_ade20k_caption_annotations()
    print("CATR Captioning", msg2)
    perform_bleu_score(captions_ade20k, caption_expert, num_ref)
    # cap = merge_annotations("/home/rafi/PycharmProjects/sose21-pm-language-and-vision-g1/annotations/captions_fully_annotated.csv", "/home/rafi/merged.csv")
