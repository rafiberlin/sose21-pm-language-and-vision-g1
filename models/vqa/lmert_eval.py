
class CustomAvatar(Avatar):
    """
        The simple avatar is only repeating the observations.
    """

    def __init__(self, image_directory):
        self.image_directory = image_directory
        self.observation = None
        self.caption_expert = get_eval_captioning_model()
        #self.vqa_expert = get_eval_vqa_model()
        #self.vqa_lxmert = get_eval_lxmert_vqa()
        self.frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
        self.image_preprocess = Preprocess(frcnn_cfg)
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.lxmert_gqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-gqa-uncased")
        self.lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json")
        with open(config_file, "r") as read_file:
            conf = json.load(read_file)["image_server"]
        self.ADE20K_URL = f"http://{conf['host']}:{conf['port']}/"

    def step(self, observation: dict) -> dict:
        print(observation)  # for debugging
        actions = dict()
        if observation["image"]:
            self.__update_observation(observation)
        if observation["message"]:
            self.__update_actions(actions, observation["message"])
        return actions

    def __update_observation(self, observation: dict):
        self.observation = observation

    def __update_actions(self, actions, message):
        if "go" in message.lower():
            actions["move"] = self.__predict_move_action(message)
        else:
            actions["response"] = self.__generate_response(message)

    def __generate_response(self, message: str) -> str:
        message = message.lower()
        image_path = None
        image_url = None
        if self.observation:
            image_url = self.ADE20K_URL + self.observation["image"]
            image_url = self.ADE20K_URL + self.observation["image"]
            last_char_index = image_url.rfind("/")
            image_name = image_url[last_char_index + 1:]
            image_path = tf.keras.utils.get_file(image_name, origin=image_url)

        if message.startswith("what"):
            if self.observation:
                caption, _ = self.caption_expert(image_path)

                return "I see ("+ image_url+"): " + ' '.join(caption[:-1])
            else:
                return "I dont know"

        if message.startswith("where"):
            if self.observation:
                return "I can go " + directions_to_sent(self.observation["directions"])
            else:
                return "I dont know"

        if message.endswith("?"):
            if self.observation:
                #answer = self.vqa_expert.infer((image_path, message))

                images, sizes, scales_yx = image_preprocess(URL)
                output_dict = frcnn(
                    images,
                    sizes,
                    scales_yx=scales_yx,
                    padding="max_detections",
                    max_detections=frcnn_cfg.max_detections,
                    return_tensors="pt")
                normalized_boxes = output_dict.get("normalized_boxes")
                features = output_dict.get("roi_features")

                inputs = lxmert_tokenizer(
                    message,
                    padding="max_length",
                    max_length=20,
                    truncation=True,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt")
                output_vqa = lxmert_vqa(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        visual_feats=features,
                        visual_pos=normalized_boxes,
                        token_type_ids=inputs.token_type_ids,
                        output_attentions=False,)

                pred_vqa = output_vqa["question_answering_score"].argmax(-1)
                return answer[0]
                # return "It has maybe something to do with " + self.observation["image"]
            else:
                return "I dont know"

        return "I do not understand"

    def __predict_move_action(self, message: str) -> str:
        if "north" in message:
            return "n"
        if "east" in message:
            return "e"
        if "west" in message:
            return "w"
        if "south" in message:
            return "s"
        return "nowhere"
