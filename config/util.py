import os
import json

def get_config():
    """
    Get the general project config
    :return:
    """
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "config.json")
    with open(config_file, "r") as read_file:
        conf = json.load(read_file)
    pretrained_root = conf["pretrained_root"]
    conf["glove_embeddings"] = os.path.join(pretrained_root,  conf["glove_embeddings"] )
    if not os.path.exists(conf["glove_embeddings"]):
        os.makedirs(conf["glove_embeddings"])
        print("Created directory:", conf["glove_embeddings"])


    conf["captioning"]["attention"]["pretrained_dir"] = os.path.join(pretrained_root,
                                            conf["captioning"]["attention"]["pretrained_dir"])

    conf["vqa"]["attention"]["pretrained_dir"] = os.path.join(pretrained_root,
                                            conf["vqa"]["attention"]["pretrained_dir"])

    captioning_pretrained_dir = conf["captioning"]["attention"]["pretrained_dir"]
    vqa_pretrained_dir = conf["vqa"]["attention"]["pretrained_dir"]

    if not os.path.exists(captioning_pretrained_dir):
        os.makedirs(captioning_pretrained_dir)
        print("Created directory:", captioning_pretrained_dir)
    if not os.path.exists(vqa_pretrained_dir):
        os.makedirs(vqa_pretrained_dir)
        print("Created directory:", vqa_pretrained_dir)
    checkpoints_dir = os.path.join(captioning_pretrained_dir, "checkpoints")
    create_directory_structure(checkpoints_dir)

    checkpoints_dir = os.path.join(vqa_pretrained_dir, "checkpoints")
    create_directory_structure(checkpoints_dir)

    checkpoints_dir = os.path.join(vqa_pretrained_dir, "logs")
    create_directory_structure(checkpoints_dir)

    if not os.path.exists(conf["game_logs_dir"]):
        os.makedirs(conf["game_logs_dir"])
        print("Created directory:", conf["game_logs_dir"])


    return conf

def create_directory_structure(struct):
    if not os.path.exists(struct):
        os.makedirs(struct)
        print("Created directory:", struct)


def read_game_logs(file_path):
    """
    It returns a dictionary where each key holds information for a particular (finished) game session.
    General statistics about the game session are provided: score, the number of questions asked by the player,
    the number of orders and whole messages during the game session
    :param file_path:
    :return:
    """

    if os.path.isfile(file_path):
        with open(file_path, "r") as read_file:
            log = json.load(read_file)
        # event_type = set([e["event"] for e in log ])
        # the event types: command, text_message, set_attribute, join
        # print("event types", event_type)

        # sort all messages chronologically
        log.sort(key=lambda x: x["date_modified"])

        start = None
        end = None

        episode_list = []
        length = len(log)
        game_finished = False
        for i, l in enumerate(log):
            if "command" in l.keys():
                if  l["command"] == "start":
                    if start == None:
                        start = i
                    elif end == None:
                        end = i
                if l["command"] == "done":
                    game_finished = True
            if start is not None and end is not None:
                if game_finished:
                    episode_list.append(log[start:end])
                start = end
                end = None
                game_finished = False

            if i + 1 == length:
                if start is not None and end is None:
                    episode_list.append(log[start:length])

        score_list = {}
        for i, e in enumerate(episode_list):
            # the number of answers the avatar utters gives us the number of question asked
            num_questions = sum(
                [1 for m in e if m["user"]["name"] == "Avatar" and m["event"] == "text_message"])

            # user id 1 is alway the game master, we are looping here on the messages of the "real" player
            # when we tell the avatar to change location, we don't get an answer, this is why the substraction gives the number of orders
            # this does not include the order "done"
            num_orders = sum(
                [1 for m in e if m["user"]["name"] != "Avatar" and m["user"]["id"] != 1 and m[
                    "event"] == "text_message"]) - num_questions

            score_list[i] = {"score": sum([m["message"]["observation"]["reward"] for m in e if
                                                              "message" in m.keys() and type(m["message"]) is dict]),
                             "num_questions": num_questions, "num_orders": num_orders, "game_session": e}

        return score_list

    else:
        raise (f"{file_path} is not a correct file path.")


if __name__ == "__main__":
    game_logs_dir = get_config()["game_logs_dir"]
    log_path = os.path.join(game_logs_dir, "rafi_10_games_04_jun_21.txt")
    log = read_game_logs(log_path)
    print(log)
