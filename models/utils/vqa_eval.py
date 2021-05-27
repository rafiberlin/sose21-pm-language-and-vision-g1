from models.utils.util import get_config, get_ade20_vqa_data
import os
import json
from models.vqa.attention_vqa import load_preprocessed_data
from models.vqa.evaluate_attention_vqa import get_eval_vqa_model
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tqdm import tqdm
import pandas as pd


def run_official_vqa_metrics():
    """
    See https://visualqa.org/evaluation.html
    :return:
    """
    # TODO Implementation is completely crap,
    # use keras model.eval with a custom callback to make the function quick to execute
    conf = get_config()

    MS_COCO_DIR = conf["ms_coco_dir"]

    _, X_val, tokenizer, label_encoder, _, question_vector_val = load_preprocessed_data()
    coco_train = os.path.join(MS_COCO_DIR, "train2017")
    # removing first and lasttoken <start>, <end>
    questions = X_val["question"].apply(lambda x: " ".join(text_to_word_sequence(x)[1:-1]))
    # we need eval here because all nested dictionaries are stored as strings...
    answers = X_val["answers"].apply(lambda x: eval(x))
    image_paths_val = X_val['image_id'].apply(lambda x: os.path.join(coco_train, '%012d.jpg' % (x))).values

    vqa = get_eval_vqa_model()
    total = 0
    for i, (question, image, _answers) in tqdm(enumerate(zip(questions, image_paths_val, answers))):
        prediction = vqa.infer((image, question))
        total += min(sum([1 for answer in _answers if answer["answer"] == prediction]) / 3, 1)
        epoch = i + 1
        if epoch % 1000 == 0:
            print("epoch", epoch, "Acc", total / epoch)
    acc = total / len(questions)
    print("VQA Accuracy", acc)

    return acc


def run_ade20k_vqa_metrics():
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", "config.json")
    with open(config_file, "r") as read_file:
        conf = json.load(read_file)

    ADE20K_DIR = conf["ade20k_dir"]

    _, X_val, tokenizer, label_encoder, _, question_vector_val = load_preprocessed_data()

    data = get_ade20_vqa_data()

    filtered_data = [d for i, d in tqdm(enumerate(data)) if d["answer"] in tokenizer.word_index.keys()]
    df = pd.DataFrame(filtered_data)
    questions = df["question"].apply(lambda x: " ".join(text_to_word_sequence(x)))
    answers = df["answer"]
    image_paths_val = df["image_path"].apply(lambda x: os.path.join(ADE20K_DIR, "images",x))
    total = 0
    vqa = get_eval_vqa_model()

    for i, (question, image, answer) in tqdm(enumerate(zip(questions, image_paths_val, answers))):
        prediction = vqa.infer((image, question))
        if prediction[0] == answer:
            total += 1
        epoch = i + 1
        if epoch % 1000 == 0:
            print("epoch", epoch, "Acc", total / epoch)
    acc = total / len(questions)
    print("ADE20K VQA Accuracy", acc)
    return acc


if __name__ == "__main__":
    run_ade20k_vqa_metrics()

    #run_official_vqa_metrics()
