from models.utils.util import get_config
import os
import json
from tqdm import tqdm
import pandas as pd
import random
import jsonlines


def read_ade20k_object_annotations():
    """
    Returns the annoations
    :return:object_annotations, image_annotations, rel_annotations
    """

    conf = get_config()
    ADE20K = conf["ade20k_dir"]
    # as found on the jarvis server under data/ImageCorpora/ADE20K_2016_07_26/preprocessed_dfs
    # with unzipped files (3 json files as a result)
    ADE20K_OBJECT_ANNOTATIONS = os.path.join(ADE20K, "preprocessed_dfs", "obj_df.json")
    with open(ADE20K_OBJECT_ANNOTATIONS, "r") as read_file:
        object_annotations = json.load(read_file)

    ADE20K_IMAGE_ANNOTATIONS = os.path.join(ADE20K, "preprocessed_dfs", "image_df.json")
    with open(ADE20K_IMAGE_ANNOTATIONS, "r") as read_file:
        image_annotations = json.load(read_file)

    ADE20K_RELATION_ANNOTATIONS = os.path.join(ADE20K, "preprocessed_dfs", "relations_df.json")
    with open(ADE20K_RELATION_ANNOTATIONS, "r") as read_file:
        rel_annotations = json.load(read_file)

    return object_annotations, image_annotations, rel_annotations


def extract_ade20k_classes(object_annotations):
    """
    attr contains the attributes of the current label
    and synset contains all synset synonyms of the current label
    :param object_annotations:
    :return: label_set, synset_set, attr_set
    """

    cols = object_annotations["columns"]
    data = object_annotations["data"]

    label_col_id = cols.index("label")
    synset_col_id = cols.index("synset")
    attr_col_id = cols.index("attr")

    label_set = set()
    synset_set = set()
    attr_set = set()

    for i, row in tqdm(enumerate(data)):

        label_set.add(row[label_col_id])
        shards = row[synset_col_id].split(",")
        if len(shards) == 1:
            synset_set.add(shards[0])
        else:
            synset_set.update(shards)
        shards = row[attr_col_id].split(",")
        if len(shards) == 1:
            attr_set.add(shards[0])
        else:
            attr_set.update(shards)

    return label_set, synset_set, attr_set


def create_ADE20K_dataset(min_labels=3):
    """
    To create questions, there must be a minimum amount of objects / labels avalaible in the picture.
    The picture will be skipped if the minimum amount is not reached
    :param min_labels:
    :return:
    """

    conf = get_config()
    ADE20K = conf["ade20k_dir"]
    VQA_FILE_NAME = conf["ade20k_vqa_file"]

    object_annotations, image_annotations, rel_annotations = read_ade20k_object_annotations()
    label_set, synset_set, attr_set = extract_ade20k_classes(object_annotations)
    label_list = list(label_set)
    obj_df = pd.DataFrame(object_annotations["data"], columns=object_annotations["columns"])
    obj_df['image_id'] = obj_df['image_id'].astype('str')
    image_df = pd.DataFrame(image_annotations["data"], columns=image_annotations["columns"])
    image_df['image_id'] = image_df['image_id'].astype('str')
    # retrieves each filepath
    merged = obj_df.merge(image_df[['image_id', 'filename', "split"]], how="left", on=["image_id", "split"])
    merged["synset"] = merged["synset"].copy().apply(lambda x: x.split(","))
    merged["attr"] = merged["attr"].copy().apply(lambda x: x.split(","))
    image_list = {f: set() for f in list(set(merged["filename"]))}
    for i, row in tqdm(merged.iterrows()):
        filename, label = row["filename"], row["label"]
        image_list[filename].add(label)

    # make results reproducible
    random.seed(0)

    question_templates = ["Is there a {} ?", "Can you see a {} ?", "Is it a {} or a {} ?", "What is it?", "What is missing {} or {} ?"]

    jsonline_path = os.path.join(ADE20K, VQA_FILE_NAME)

    with jsonlines.open(jsonline_path, 'w') as f_out:
        for key in tqdm(image_list.keys()):
            val = list(image_list[key])
            if len(val) >= min_labels:
                positive_examples = random.sample(val, k=min_labels)
                negative_examples = random.sample([s for s in label_list if s not in val], k=min_labels)
                questions = []
                answers = []
                for p in positive_examples:
                    questions.append(question_templates[0].format(p))
                    answers.append("yes")
                    questions.append(question_templates[1].format(p))
                    answers.append("yes")
                    n = random.choice(negative_examples)
                    questions.append(question_templates[2].format(p, n))
                    answers.append(p)
                    questions.append(question_templates[3])
                    answers.append(p)
                for neg in negative_examples:
                    questions.append(question_templates[0].format(neg))
                    answers.append("no")
                    questions.append(question_templates[1].format(neg))
                    answers.append("no")
                    n = random.sample(negative_examples, k=2)
                    questions.append(question_templates[2].format(n[0], n[1]))
                    answers.append("none")
                    pos = random.sample(positive_examples, k=1)
                    questions.append(question_templates[4].format(pos[0], neg))
                    answers.append(neg)
                for q, a, in zip(questions, answers):
                    f_out.write({"image_path": key, "question": q, "answer": a})


if __name__ == "__main__":
    create_ADE20K_dataset()
