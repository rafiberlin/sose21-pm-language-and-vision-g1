import os
import json


def get_config():
    """
    Get the general project config
    :return:
    """
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_file, "r") as read_file:
        conf = json.load(read_file)

    if conf["use_dev_config"]:
        print("Dev Setup: dev_config.json will be used")
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev_config.json")
        with open(config_file, "r") as read_file:
            conf = json.load(read_file)
    else:
        print("Server Setup: config.json will be used")

    pretrained_root = conf["pretrained_root"]
    gloves_dir = os.path.join(pretrained_root, "gloves")
    if not os.path.exists(gloves_dir):
        os.makedirs(gloves_dir)
        print("Created directory:", gloves_dir)

    conf["captioning"]["attention"]["pretrained_dir"] = os.path.join(pretrained_root,
                                                                     conf["captioning"]["attention"]["pretrained_dir"])

    conf["vqa"]["attention"]["pretrained_dir"] = os.path.join(pretrained_root,
                                                              conf["vqa"]["attention"]["pretrained_dir"])

    conf["vqa"]["lxmert"]["fine_tuning"]["pretrained_dir"] = os.path.join(pretrained_root,
                                                                          conf["vqa"]["lxmert"]["fine_tuning"][
                                                                              "pretrained_dir"])

    captioning_pretrained_dir = conf["captioning"]["attention"]["pretrained_dir"]
    vqa_pretrained_dir = conf["vqa"]["attention"]["pretrained_dir"]
    lxmert_pretrained_dir = conf["vqa"]["lxmert"]["fine_tuning"]["pretrained_dir"]

    vqa_features = os.path.join(conf["ade20k_vqa_dir"], "precomputed_features/training")

    gqa_dir = os.path.join(lxmert_pretrained_dir,
                           "gqa")
    vqa_dir = os.path.join(lxmert_pretrained_dir,
                           "vqa")

    if not os.path.exists(captioning_pretrained_dir):
        os.makedirs(captioning_pretrained_dir)
        print("Created directory:", captioning_pretrained_dir)
    if not os.path.exists(vqa_pretrained_dir):
        os.makedirs(vqa_pretrained_dir)
        print("Created directory:", vqa_pretrained_dir)

    if not os.path.exists(lxmert_pretrained_dir):
        os.makedirs(lxmert_pretrained_dir)
        print("Created directory:", lxmert_pretrained_dir)

    if not os.path.exists(gqa_dir):
        os.makedirs(gqa_dir)
        print("Created directory:", gqa_dir)

    if not os.path.exists(vqa_dir):
        os.makedirs(vqa_dir)
        print("Created directory:", vqa_dir)

    if not os.path.exists(vqa_features):
        os.makedirs(vqa_features)
        print("Created directory:", vqa_features)

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
        real_end = None  # WHen The came master says COngrats or you die, because rest of the messages looks like bugs...
        episode_list = []
        length = len(log)
        game_finished = False
        # Episode are being searched between 2 starts commands
        # only the one where the command done has been issued is kept
        for i, l in enumerate(log):
            if "command" in l.keys():
                if l["command"] == "start":
                    if start == None:
                        start = i
                    elif end == None:
                        end = i
                if l["command"] == "done":
                    game_finished = True

            if l["user"]["id"] == 1 and l["event"] == "text_message" and type(l["message"]) is str and (
                    l["message"].startswith("Congrats") or l["message"].startswith(
                "The rescue robot has not reached you")):
                real_end = i + 1  # +1 because we want to include this message in the log slice...
            if start is not None and end is not None:
                if game_finished:
                    episode_list.append(log[start:real_end])
                start = end
                end = None
                real_end = None
                game_finished = False

            if i + 1 == length:
                if start is not None and end is None and game_finished:
                    episode_list.append(log[start:real_end])

        score_list = {}
        for i, e in enumerate(episode_list):
            # the number of answers the avatar utters gives us the number of question asked
            # num_questions = sum(
            #     [1 for m in e if m["user"]["name"] == "Avatar" and m["event"] == "text_message"])

            # Just sum every messages ending with a question mark issueed by the user...
            num_questions = sum([1 for m in e if m["user"]["name"] != "Avatar" and m["user"]["id"] != 1 and m[
                "event"] == "text_message" and type(m["message"]) is str and m["message"].endswith("?")])

            # user id 1 is alway the game master, we are looping here on the messages of the "real" player
            # when we tell the avatar to change location, we don't get an answer, this is why the substraction gives the number of orders
            # this does not include the order "done"
            # num_orders = sum(
            #     [1 for m in e if m["user"]["name"] != "Avatar" and m["user"]["id"] != 1 and m[
            #         "event"] == "text_message"]) - num_questions

            # Just sum every order of type "go west". Describe orders are not counted.
            num_orders = sum([1 for m in e if m["user"]["name"] != "Avatar" and m["user"]["id"] != 1 and m[
                "event"] == "text_message" and type(m["message"]) is str and (
                                      "east" in m["message"].lower() or "north" in m["message"].lower() or "west" in m[
                                  "message"].lower() or "south" in m["message"].lower() or "back" in m["message"].lower())])

            game_won = sum([1 for m in e if m["user"]["id"] == 1 and m[
                "event"] == "text_message" and type(m["message"]) is str and m["message"].startswith("Congrats")]) > 0

            score_list[i] = {"score": sum([m["message"]["observation"]["reward"] for m in e if
                                           "message" in m.keys() and type(m["message"]) is dict]),
                             "num_questions": num_questions, "num_orders": num_orders, "game_session": e,
                             "game_won": game_won}

        return score_list

    else:
        raise Exception(f"{file_path} is not a correct file path.")


def output_game_metrics(log):
    num_game = len(log)
    PENALTY_FOR_QUESTION_ASKED = -0.01
    discounted_score = lambda l, i: l[i]["score"] + l[i]["num_questions"] * PENALTY_FOR_QUESTION_ASKED

    s = 0
    sq = 0
    for k in log.keys():
        sq += discounted_score(log, k)
        s += log[k]["score"]

    print("Average Score", s / num_game)
    print("Won Games", f"{sum([1 for k in log.keys() if log[k]['game_won']])} / {num_game}")
    print("Questions Asked Per Game", f"{sum([log[k]['num_questions'] for k in log.keys()]) / num_game}")
    print("Orders Given Per Game", f"{sum([log[k]['num_orders'] for k in log.keys()]) / num_game}")

    max_score_id = max(log.keys(), key=lambda k: log[k]["score"])
    print("Best Game under normal Score", log[max_score_id]["score"], "Game Number",
          max_score_id)
    lowest_score_id = min(log.keys(), key=lambda k: log[k]["score"])
    print("Worse Game under normal Score", log[lowest_score_id]["score"], "Game Number",
          lowest_score_id)
    print("Average Score with questions discount", sq / num_game)

    max_score_question_discount_id = max(log.keys(), key=lambda k: discounted_score(log, k))

    print("Best Game with Question Penalty Score",
          discounted_score(log, max_score_question_discount_id), "Game number",
          max_score_question_discount_id)
    min_score_question_discount_id = min(log.keys(), key=lambda k: discounted_score(log, k))
    print("Worse Game under Question Penalty Score",
          discounted_score(log, min_score_question_discount_id), "Game Number",
          min_score_question_discount_id)

    return (s, sq)


if __name__ == "__main__":
    print("Executing the script once should create every directories needed for proper execution.")
    get_config()
    print("Done")
