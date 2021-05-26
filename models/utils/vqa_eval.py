from models.utils.util import get_config
import os
import json
from models.vqa.attention_vqa import load_preprocessed_data
from models.vqa.attention_vqa import create_dataset
from models.vqa.evaluate_attention_vqa import  get_eval_vqa_model
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tqdm import tqdm
def read_ade20k_object_annotations():
    conf = get_config()
    ADE20K = conf["ad20k_dir"]
    # as found on the jarvis server under data/ImageCorpora/ADE20K_2016_07_26/preprocessed_dfs
    # with unzipped files (3 json files as a result)
    ADE20K_OBJECT_ANNOTATIONS = os.path.join(ADE20K, "preprocessed_dfs", "obj_df.json")
    with open(ADE20K_OBJECT_ANNOTATIONS, "r") as read_file:
        object_annotations = json.load(read_file)

    return object_annotations

def run_basic_vqa_on_ade20k():
    object_annotations = read_ade20k_object_annotations()


    pass

def run_official_vqa_metrics():
    """
    See https://visualqa.org/evaluation.html
    :return:
    """
    #TODO Implementation is completely crap,
    # use keras model.eval with a custom callback to make the function quick to execute
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", "config.json")
    with open(config_file, "r") as read_file:
        conf = json.load(read_file)

    MS_COCO_DIR = conf["ms_coco_dir"]

    _, X_val, tokenizer, label_encoder, _, question_vector_val = load_preprocessed_data()
    coco_train = os.path.join(MS_COCO_DIR, "train2017")

    questions = X_val["question"].apply(lambda x: " ".join(text_to_word_sequence(x)[1:-1]))
    answers = X_val["answers"].apply(lambda x: eval(x))
    image_paths_val = X_val['image_id'].apply(lambda x: os.path.join(coco_train, '%012d.jpg' % (x))).values

    vqa = get_eval_vqa_model()
    total = 0
    for i, (question, image, answers) in tqdm(enumerate(zip(questions, image_paths_val, answers))):
        prediction = vqa.infer(("file://"+image, question))
        total += min(sum([ 1 for answer in answers if answer["answer"] == prediction])/3, 1)
        epoch = i+ 1
        if epoch % 1000 == 0:
            print("epoch", epoch,"Acc", total/ epoch)
    acc = total / len(questions)
    print("VQA Accuracy", acc)

    keras_cache = os.path.join(os.path.expanduser("~/"), ".keras/datasets")
    filelist = [f for f in os.listdir(keras_cache)]
    for f in filelist:
        os.remove(os.path.join(keras_cache, f))


    return acc

if __name__ == "__main__":

    run_official_vqa_metrics()
