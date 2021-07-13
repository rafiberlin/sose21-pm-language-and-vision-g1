import asyncio
from rasa.core.agent import Agent
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

    def __init__(self, image_directory, caption_expert, vqa_expert):
        assert image_directory is not None
        assert caption_expert is not None
        assert vqa_expert is not None
        if not image_directory.endswith("/"):
            image_directory = image_directory + "/"
        self.image_directory = image_directory
        self.observation = None
        self.caption_expert = caption_expert
        self.vqa_expert = vqa_expert
        conf = get_config()
        self.debug = conf["image_server"]["debug"]
        self.rasa_model_path = conf["rasa"]["model_dir"]
        self.device = conf["rasa"]["tensorflow_gpu_name"]
        self.previous = []
        self.agent = Agent.load(self.rasa_model_path)

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
        if not "where" in message.lower() and "go" in message.lower() or not "where" in message.lower() and "move" in message.lower():
            direction = self.__predict_move_action(message.lower())
            if "back" in message.lower() or "previous" in message.lower():
                # last element of the list is the previous direction
                direction = self.__go_back(self.previous[-1])
            actions["move"] = direction
            self.previous = direction
        else:
            actions["response"] = self.__generate_response(message)

    def __generate_response(self, message: str) -> str:
        message = message.lower()
        response = asyncio.run(self.agent.handle_text(message))[0]['text']
        image_path= None
        image_url = None

        if self.observation:
            image_path, image_url = self.__get_image_path_and_url(self.observation)

        if response == "get_caption_expert_answer":
            if self.observation:
                caption = self.caption_expert.infer(image_path)
                return self.__format_caption_answer(caption, image_url)
            else:
                return "I dont know"

        if response == "get_possible_directions":
            if self.observation:
                return "I can go " + directions_to_sent(self.observation["directions"])
            else:
                return "I dont know"

        if response == "get_vqa_expert_answer":
            if self.observation:
                answer = self.vqa_expert.infer(image_path, message)
                return answer
            else:
                return "I dont know"

        return response

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

    def __go_back(self, direction: str) -> str:
        if direction == "n":
            return "s"
        if direction == "s":
            return "n"
        if direction == "w":
            return "e"
        if direction == "e":
            return "w"

    def __get_image_path_and_url(self, observation):
        image_path = None
        image_url = None
        if observation["image"]:
            if self.observation:
                image_url = self.image_directory + self.observation["image"]
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