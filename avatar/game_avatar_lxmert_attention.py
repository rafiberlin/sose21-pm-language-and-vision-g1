from avatar.game_avatar import Avatar
from avatar_models.captioning.evaluate import get_eval_captioning_model
from avatar_models.vqa.lxmert.lxmert import LXMERTInference
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

class LXMERTAttentionAvatar(Avatar):
#        The simple avatar is only repeating the observations.

    def __init__(self, image_directory):
        self.image_directory = image_directory
        self.observation = None
        self.caption_expert = get_eval_captioning_model()
        #self.vqa_expert = get_eval_vqa_model()
        self.vqa_expert = LXMERTInference()
        conf = get_config()
        conf = conf["image_server"]
        self.ADE20K_URL = f"http://{conf['host']}:{conf['port']}/"
        self.debug = conf["debug"]

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
            last_char_index = image_url.rfind("/")
            image_name = image_url[last_char_index + 1:]
            image_path = tf.keras.utils.get_file(image_name, origin=image_url)

        if message.startswith("describe"):
            if self.observation:
                caption, _ = self.caption_expert(image_path)
                debug_msg = ""
                if self.debug:
                    debug_msg = f'("+ {image_url}+")'
                return f"I see {debug_msg}: " + ' '.join(caption[:-1])

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

if __name__ == "__main__":
    #URL = "https://vignette.wikia.nocookie.net/spongebob/images/2/20/SpongeBob's_pineapple_house_in_Season_7-4.png/revision/latest/scale-to-width-down/639?cb=20151213202515"
    URL =  "https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Ffeedinspiration.com%2Fwp-content%2Fuploads%2F2016%2F03%2FTransitional-Kitchen-Cabinets-Ideas.jpg&f=1&nofb=1"
    test_question = "what is this?"
    vqa_model = LXMERTInference()
    vqa_answer = vqa_model.infer(URL, test_question)
    print(vqa_answer)
