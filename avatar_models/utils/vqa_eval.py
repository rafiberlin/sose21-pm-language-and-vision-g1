from avatar_models.utils.util import get_config, get_ade20_vqa_data, load_preprocessed_vqa_data
import os
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tqdm import tqdm
import pandas as pd
from avatar_models.vqa.evaluate_attention_vqa import get_eval_vqa_model
from avatar_models.vqa.lxmert.lxmert import LXMERTInference
import numpy as np
import time

def run_official_vqa_metrics(vqa, answer_list, num_questions=None):
    """
    See https://visualqa.org/evaluation.html
    :return:
    """
    # TODO Implementation is completely crap,
    # use keras model.eval with a custom callback to make the function quick to execute
    conf = get_config()
    _, X_val, _, _, _, _ = load_preprocessed_vqa_data()
    MS_COCO_DIR = conf["ms_coco_dir"]
    derived_answer_col_id = np.where(X_val.columns.values == "derived_answer")[0]
    coco_train = os.path.join(MS_COCO_DIR, "train2017")
    filtered = pd.DataFrame(
        [row for row in X_val.itertuples(index=False) if row[int(derived_answer_col_id)] in answer_list])
    # removing first and lasttoken <start>, <end>
    questions = filtered["question"].apply(lambda x: " ".join(text_to_word_sequence(x)[1:-1]))
    # we need eval here because all nested dictionaries are stored as strings...
    answers = filtered["answers"].apply(lambda x: eval(x))
    image_paths_val = filtered['image_id'].apply(lambda x: os.path.join(coco_train, '%012d.jpg' % (x))).values

    print(f"Dataset contains {len(X_val)} questions")
    print(f"After filtering (keeping only answers in the model vocabulary), test on {len(answers)} questions")

    total = 0
    epoch = 0
    for i, (question, image, _answers) in tqdm(enumerate(zip(questions, image_paths_val, answers))):
        prediction = vqa.infer(image, question)
        total += min(sum([1 for answer in _answers if answer["answer"] == prediction]) / 3, 1)
        epoch += 1
        # Work around to avoid reboots on my old GPU...
        if epoch % 500 == 0:
            time.sleep(30)
        if epoch % 1000 == 0:
            print("epoch", epoch, "Acc", total / epoch)
        if epoch == num_questions:
            break
    acc = total / epoch
    print("MSCOCO VQA Accuracy", acc, f"{epoch} questions tested")

    return acc


def run_ade20k_vqa_metrics(vqa, answer_list, num_questions=None):
    conf = get_config()
    ADE20K_DIR = conf["ade20k_dir"]
    VQA_FILE = conf["ade20k_vqa_file_test"]
    data = get_ade20_vqa_data(VQA_FILE)

    filtered_data = [d for i, d in tqdm(enumerate(data)) if d["answer"] in answer_list]
    df = pd.DataFrame(filtered_data)
    questions = df["question"].apply(lambda x: " ".join(text_to_word_sequence(x)))
    answers = df["answer"]
    image_paths_val = df["image_path"].apply(lambda x: os.path.join(ADE20K_DIR, "images", x))
    total = 0
    epoch = 0

    for i, (question, image, answer) in tqdm(enumerate(zip(questions, image_paths_val, answers))):
        prediction = vqa.infer(image, question)
        if prediction == answer:
            total += 1
        epoch +=  1
        #Work around to avoid reboots on my old GPU...
        if epoch % 500 == 0:
            time.sleep(30)

        if epoch % 1000 == 0:
            print("epoch", epoch, "Acc", total / epoch)
        if epoch == num_questions:
            break  # the LXMERT is really slow at processing, it would take ages...
    acc = total / epoch
    print("ADE20K VQA Accuracy", acc, f"{epoch} questions tested")
    return acc


if __name__ == "__main__":
    _, _, tokenizer, _, _, _ = load_preprocessed_vqa_data()
    max_questions = get_config()["ade20k_vqa_max_questions"]
    vqa_attention = get_eval_vqa_model()
    answer_vocab__attention = tokenizer.word_index.keys()
    print("##########  START : Run VQA Test on Attention Model ##########")
    print("MSCOCO Dataset")
    run_official_vqa_metrics(vqa_attention, answer_vocab__attention, max_questions)
    print("ADE20K Dataset")
    run_ade20k_vqa_metrics(vqa_attention, answer_vocab__attention, max_questions)
    print("##########  END : Run VQA Test on Attention Model ##########")

    vqa_lxmert = LXMERTInference()
    answer_vocab_lxmert = vqa_lxmert.get_answers()
    print(f"##########  START : Run VQA Test on LXMERT Model {vqa_lxmert.model_type}##########")
    print("MSCOCO Dataset")
    run_official_vqa_metrics(vqa_lxmert, answer_vocab_lxmert, max_questions)
    print("ADE20K Dataset")
    run_ade20k_vqa_metrics(vqa_lxmert, answer_vocab_lxmert, max_questions)
    print(f"##########  END : Run VQA Test on LXMERT Model {vqa_lxmert.model_type}##########")
