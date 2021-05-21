from models.utils.util import get_config
import pandas as pd
import os
from models.captioning.evaluate import get_eval_captioning_model
from pycocoevalcap.bleu.bleu import Bleu
from nltk.tokenize import word_tokenize
from tqdm import tqdm
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

if __name__ == "__main__":
    caption = get_ade20k_caption_annotations()
    caption_expert = get_eval_captioning_model()
    scores = []
    references = {}
    hypothesis = {}
    for i, row in tqdm(caption.iterrows()):
        gold_caption = " ".join(word_tokenize(row["caption"]))
        image_path = row["image_path"]
        image_id = row["image_id"]
        predicted_caption, _ = caption_expert(image_path)
        if predicted_caption[-1] == "<end>":
            predicted_caption = predicted_caption[:-1]
        predicted_caption = " ".join(predicted_caption)
        references[image_id]= [gold_caption]
        hypothesis[image_id] = [predicted_caption]


    scores = calc_scores(references, hypothesis)
    print(scores)