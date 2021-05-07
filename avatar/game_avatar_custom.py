from avatar.game_avatar import Avatar
from models.captioning import evaluate
import tensorflow as tf
import json
import os
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
    """
        The simple avatar is only repeating the observations.
    """

    def __init__(self, image_directory):
        self.image_directory = image_directory
        self.observation = None
        self.caption_expert = evaluate.get_eval_model()
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

        if message.startswith("what"):
            if self.observation:

                image_url = self.ADE20K_URL+self.observation["image"]
                last_char_index = image_url.rfind("/")
                image_name = image_url[last_char_index + 1:]
                image_path = tf.keras.utils.get_file(image_name, origin=image_url)

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
                return "It has maybe something to do with " + self.observation["image"]
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
