import os
import json

def get_config():
    """
    Get the general project config
    :return:
    """
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "config.json")
    with open(config_file, "r") as read_file:
        conf = json.load(read_file)
    pretrained_root = conf["pretrained_root"]
    conf["glove_embeddings"] = os.path.join(pretrained_root,  conf["glove_embeddings"] )
    if not os.path.exists(conf["glove_embeddings"]):
        os.makedirs(conf["glove_embeddings"])
        print("Created directory:", conf["glove_embeddings"])


    conf["captioning"]["pretrained_dir"] = os.path.join(pretrained_root,
                                            conf["captioning"]["pretrained_dir"])

    conf["vqa"]["attention"]["pretrained_dir"] = os.path.join(pretrained_root,
                                            conf["vqa"]["attention"]["pretrained_dir"])

    captioning_pretrained_dir = conf["captioning"]["pretrained_dir"]
    vqa_pretrained_dir = conf["vqa"]["attention"]["pretrained_dir"]

    if not os.path.exists(captioning_pretrained_dir):
        os.makedirs(captioning_pretrained_dir)
        print("Created directory:", captioning_pretrained_dir)
    if not os.path.exists(vqa_pretrained_dir):
        os.makedirs(vqa_pretrained_dir)
        print("Created directory:", vqa_pretrained_dir)
    checkpoints_dir = os.path.join(captioning_pretrained_dir, "checkpoints")
    create_directory_structure(checkpoints_dir)

    checkpoints_dir = os.path.join(vqa_pretrained_dir, "checkpoints")
    create_directory_structure(checkpoints_dir)

    checkpoints_dir = os.path.join(vqa_pretrained_dir, "logs")
    create_directory_structure(checkpoints_dir)

    return conf

def create_directory_structure(struct):
    if not os.path.exists(struct):
        os.makedirs(struct)
        print("Created directory:", struct)


if __name__ == "__main__":

    #perform_bleu_score_on_ade20k()
    c = get_config()
    pass