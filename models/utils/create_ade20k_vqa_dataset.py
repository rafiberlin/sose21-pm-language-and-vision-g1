from models.utils.util import get_config
import os
import json
from tqdm import tqdm
import pandas as pd

def read_ade20k_object_annotations():
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
    cols = object_annotations["columns"]
    data = object_annotations["data"]

    label_col_id = cols.index("label")
    synset_col_id = cols.index("synset")
    attr_col_id = cols.index("attr")

    label_set =  set()
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


def create_ADE20K_dataset():
    object_annotations, image_annotations, rel_annotations  = read_ade20k_object_annotations()
    label_set, synset_set, attr_set = extract_ade20k_classes(object_annotations)
    obj_df = pd.DataFrame(object_annotations["data"], columns=object_annotations["columns"])
    obj_df['image_id'] = obj_df['image_id'].astype('str')
    image_df = pd.DataFrame(image_annotations["data"], columns=image_annotations["columns"])
    image_df['image_id'] = image_df['image_id'].astype('str')
    #retrieves each filepath
    merged = obj_df.merge(image_df[['image_id', 'filename', "split"]], how="left", on=["image_id", "split"])
    merged["synset"] = merged["synset"].copy().apply(lambda x: x.split(","))
    merged["attr"] = merged["attr"].copy().apply(lambda x: x.split(","))
    pass


if __name__ == "__main__":

    create_ADE20K_dataset()

