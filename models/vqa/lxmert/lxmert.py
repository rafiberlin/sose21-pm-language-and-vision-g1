from IPython.display import clear_output, Image, display
import PIL.Image
import sys
#sys.path.append("/home/kev/sose21-pm-language-and-vision-g1/models/vqa/lxmert/")
import io
import json
import torch
import numpy as np
from models.vqa.lxmert.processing_image import Preprocess
from models.vqa.lxmert.visualizing_image import SingleImageViz
from models.vqa.lxmert.modeling_frcnn import GeneralizedRCNN
from models.vqa.lxmert.utils import Config, get_data

from transformers import LxmertForQuestionAnswering, LxmertTokenizer
import wget
import pickle
import os



def infer_lxmert_vqa (URL, test_question):
    #URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/images/input.jpg",
    #URL = "https://vqa.cloudcv.org/media/test2014/COCO_test2014_000000262567.jpg"
    OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
    ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
    #GQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/gqa/trainval_label2ans.json"
    VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"

    """
    # for visualizing output
    def showarray(a, fmt='jpeg'):
        a = np.uint8(np.clip(a, 0, 255))
        f = io.BytesIO()
        PIL.Image.fromarray(a).save(f, fmt)
        display(Image(data=f.getvalue()))
    """
    # load object, attribute, and answer labels
    objids = get_data(OBJ_URL)
    attrids = get_data(ATTR_URL)
    #gqa_answers = utils.get_data(GQA_URL)
    vqa_answers = get_data(VQA_URL)

    # load models and model components
    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")

    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)

    image_preprocess = Preprocess(frcnn_cfg)

    lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    #lxmert_gqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-gqa-uncased")
    lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")

    #image viz
    frcnn_visualizer = SingleImageViz(URL, id2obj=objids, id2attr=attrids)
    # run frcnn
    images, sizes, scales_yx = image_preprocess(URL)
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt"
    )
    # add boxes and labels to the image

    frcnn_visualizer.draw_boxes(
        output_dict.get("boxes"),
        output_dict.pop("obj_ids"),
        output_dict.pop("obj_probs"),
        output_dict.pop("attr_ids"),
        output_dict.pop("attr_probs"),
    )
    #showarray(frcnn_visualizer._get_buffer())

    #Very important that the boxes are normalized
    normalized_boxes = output_dict.get("normalized_boxes")
    features = output_dict.get("roi_features")


    inputs = lxmert_tokenizer(
        test_question,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    """
        # run lxmert(s)
        output_gqa = lxmert_gqa(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=inputs.token_type_ids,
            output_attentions=False,
        )

    """
    output_vqa = lxmert_vqa(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        visual_feats=features,
        visual_pos=normalized_boxes,
        token_type_ids=inputs.token_type_ids,
        output_attentions=False,
    )
    # get prediction
    pred_vqa = output_vqa["question_answering_score"].argmax(-1)
#    pred_gqa = output_gqa["question_answering_score"].argmax(-1)
#    print("Question:", test_question)
#    print("prediction from LXMERT GQA:", gqa_answers[pred_gqa])
#    print("prediction from LXMERT VQA:", vqa_answers[pred_vqa])
    print(vqa_answers[pred_vqa])
    return vqa_answers[pred_vqa]
if __name__ == "__main__":
    #URL = "https://www.wallpapers13.com/wp-content/uploads/2015/12/Nature-Lake-Bled.-Desktop-background-image.jpg"
    #URL = "https://vignette.wikia.nocookie.net/spongebob/images/2/20/SpongeBob's_pineapple_house_in_Season_7-4.png/revision/latest/scale-to-width-down/639?cb=20151213202515"
    URL = "https://www.quizible.com/sites/quiz/files/imagecache/question/quiz/pictures/2012/06/14/q40716.jpg"
    test_question = "which animal is this?"
    infer_lxmert_vqa(URL, test_question)
