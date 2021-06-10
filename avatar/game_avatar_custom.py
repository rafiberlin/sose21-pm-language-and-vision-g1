from avatar.game_avatar import Avatar
import tensorflow as tf
from config.util import get_config


"""
    Avatar action routines

"""
DIRECTION_TO_WORD = {
    "n": "north",
    "e": "east",
    "w": "west",
    "s": "south"
}

ADE20K_URL = "http://localhost:8000/"

def direction_to_word(direction: str):
    if direction in DIRECTION_TO_WORD:
        return DIRECTION_TO_WORD[direction]
    return direction


def directions_to_sent(directions: str):
    if not directions:
        return "nowhere"
    n = len(directions)
    if n == 1:
        return direction_to_word(directions[0])
    words = [direction_to_word(d) for d in directions]
    return ", ".join(words[:-1]) + " or " + words[-1]

class CustomAvatar(Avatar):
#        The simple avatar is only repeating the observations.

    def __init__(self, image_directory, caption_expert, vqa_expert):
        self.image_directory = image_directory
        self.observation = None
        self.caption_expert = caption_expert
        #self.vqa_expert = get_eval_vqa_model()
        self.vqa_expert = vqa_expert
        conf = get_config()
        conf = conf["image_server"]
        self.ADE20K_URL = f"http://{conf['host']}:{conf['port']}/"
        self.debug = conf["debug"]

    def step(self, observation: dict) -> dict:
        print(observation)  # for debugging
        actions = dict()
        if observation["image"]:
            self.__update_observation(observation)
            actions["response"] = self.__get_caption_expert_answer(observation)


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
        image_path= None
        image_url = None
        if self.observation:
            image_path, image_url = self.__get_image_path_and_url(self.observation)

        if message.startswith("describe"):
            if self.observation:
                caption = self.caption_expert.infer(image_path)
                return self.__format_caption_answer(caption, image_url)

            else:
                return "I dont know"

        if message.startswith("where"):
            if self.observation:
                return "I can go " + directions_to_sent(self.observation["directions"])
            else:
                return "I dont know"

        if message.endswith("?"):
            if self.observation:
                answer = self.vqa_expert.infer(image_path, message)
                return answer
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
    def __get_image_path_and_url(self, observation):
        image_path = None
        image_url = None
        if observation["image"]:
            if self.observation:
                image_url = self.ADE20K_URL + self.observation["image"]
                last_char_index = image_url.rfind("/")
                image_name = image_url[last_char_index + 1:]
                image_path = tf.keras.utils.get_file(image_name, origin=image_url)
        return image_path, image_url

    def __get_caption_expert_answer(self, observation):
        image_path, image_url = self.__get_image_path_and_url(observation)

        caption = self.caption_expert.infer(image_path)
        return self.__format_caption_answer(caption, image_url)

    def __format_caption_answer(self, caption, image_url):
        debug_msg = ""
        if self.debug:
            debug_msg = f'("+ {image_url}+")'
        return f"I see {debug_msg}: " + caption

