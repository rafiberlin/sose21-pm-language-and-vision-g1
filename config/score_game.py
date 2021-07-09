import os
from config.util import get_config, read_game_logs, output_game_metrics
import argparse

# Execute as score_game.py --file rafi_10_games_04_jun_21_attention_caption_lxmert_vqa.txt --dir ../data/game_logs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate the game statistics given a game log file')
    parser.add_argument("--file", help="the file to process", required=True)
    parser.add_argument("--dir",
                        help="The directory where to find the file. If not given, it will use the configured "
                             "path under 'game_logs_dir' in the  config file",
                        default=None)
    args = parser.parse_args()
    file_name = args.file
    game_logs_dir = args.dir
    if game_logs_dir is None:
        game_logs_dir = get_config()["game_logs_dir"]
    log_path = os.path.join(game_logs_dir, file_name)
    log = read_game_logs(log_path)
    output_game_metrics(log)
