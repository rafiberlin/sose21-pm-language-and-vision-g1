from avatar_models.vqa.lxmert.processing_image import Preprocess
from avatar_models.vqa.lxmert.visualizing_image import SingleImageViz
from avatar_models.vqa.lxmert.modeling_frcnn import GeneralizedRCNN
from avatar_models.vqa.lxmert.utils import Config, get_data
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
from avatar_models.utils.util import get_config
import os
import torch


class LXMERTInference():

    def __init__(self, model_type=None):
        """
        Create a LXMERT inference Model
        :param model_type: String, either vqa or gqa
        """

        self.OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
        self.ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"

        conf = get_config()
        lxmert_conf = conf["vqa"]["lxmert"]
        device = conf["vqa"]["lxmert"]["cuda_device"]

        if model_type is None:
            self.model_type = lxmert_conf["model"]
        else:
            self.model_type = model_type

        if type(self.model_type) is str:
            self.model_type = self.model_type.lower()

        if self.model_type == "gqa":
            print("Loading GQA Model for LXMERT")
            self.MODEL_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/gqa/trainval_label2ans.json"
            self.vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-gqa-uncased")
        else:
            print("Loading default VQA Model for LXMERT")
            self.model_type = "vqa"
            self.MODEL_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"
            self.vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")

        if lxmert_conf["fine_tuning"]["use_pretrained"]:
            fine_tuned_model = os.path.join(conf["pretrained_root"], lxmert_conf["fine_tuning"]["pretrained_dir"], self.model_type, lxmert_conf["fine_tuning"]["model_file"])
            if not os.path.exists(fine_tuned_model):
                raise Exception(f"{fine_tuned_model} does not exit, VQA LXMERT with fine tuning cannot be loaded")
            print(f"Fine-tuned weights for VQA loaded: {fine_tuned_model}")
            ckpt = torch.load(fine_tuned_model, map_location=device)
            self.vqa.load_state_dict(ckpt["state_dict"])

        self.QUESTION_LENGTH = lxmert_conf["question_length"]
        """
        # for visualizing output
        def showarray(a, fmt='jpeg'):
            a = np.uint8(np.clip(a, 0, 255))
            f = io.BytesIO()
            PIL.Image.fromarray(a).save(f, fmt)
            display(Image(data=f.getvalue()))
        """
        # load object, attribute, and answer labels
        self.objids = get_data(self.OBJ_URL)
        self.attrids = get_data(self.ATTR_URL)
        # gqa_answers = utils.get_data(GQA_URL)
        self.answers = get_data(self.MODEL_URL)

        # load avatar_models and model components
        self.frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")

        if torch.cuda.is_available() and device is not None and device.startswith("cuda"):
            print("Enabling CUDA for LXMERT", device)
            self.frcnn_cfg.model.device = device



        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)

        self.image_preprocess = Preprocess(self.frcnn_cfg)

        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")


    def get_answers(self):
        """
        Returns a list of the possible model answers.
        :return: a list of answers
        """
        return self.answers

    # run frcnn
    @torch.no_grad()
    def infer(self, URL, test_question ):
        """
        Given an image url and a question, returns an answer.
        :param URL: image stored at the url
        :param test_question: a question concerning the image
        :return: an answer
        """

        # URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/images/input.jpg",
        # URL = "https://vqa.cloudcv.org/media/test2014/COCO_test2014_000000262567.jpg"
        # image viz
        frcnn_visualizer = SingleImageViz(URL, id2obj=self.objids, id2attr=self.attrids)

        images, sizes, scales_yx = self.image_preprocess(URL)
        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=self.frcnn_cfg.max_detections,
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
        # showarray(frcnn_visualizer._get_buffer())

        # Very important that the boxes are normalized
        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")

        inputs = self.lxmert_tokenizer(
            test_question,
            padding="max_length",
            max_length=self.QUESTION_LENGTH,
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
        output_vqa = self.vqa(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=inputs.token_type_ids,
            output_attentions=False
        )
        # get prediction
        pred_vqa = output_vqa["question_answering_score"].argmax(-1)
        #    pred_gqa = output_gqa["question_answering_score"].argmax(-1)
        #    print("Question:", test_question)
        #    print("prediction from LXMERT GQA:", gqa_answers[pred_gqa])
        #    print("prediction from LXMERT VQA:", vqa_answers[pred_vqa])
        #print(vqa_answers[pred_vqa])
        return self.answers[pred_vqa]


