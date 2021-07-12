# A MapWorld Avatar Game (SoSe21 PM Vision)

# Pretrained Models

## Captioning

### Attention Model

#### MSCOCO Dataset

In the configuration file ./config.json, the key "ms_coco_dir" indicates the location of the MSCOCO dataset,
which is required to train the captioning model.

Let's call the directory defined for the key "ms_coco_dir" [MS_COCO_DIR]

Download the following files and unzip them under [MS_COCO_DIR]:

http://images.cocodataset.org/zips/train2017.zip

http://images.cocodataset.org/zips/val2017.zip

http://images.cocodataset.org/annotations/annotations_trainval2017.zip

This would result in 3 new directories containing images and annotations:

[MS_COCO_DIR]/train2017
[MS_COCO_DIR]/val2017
[MS_COCO_DIR]/annotations

#### Glove embeddings (optional)

The glove embeddings can be used.

Download the files from http://nlp.stanford.edu/data/glove.6B.zip

Unzip the content under ./pretrained/gloves

The location for the downloaded Glove embeddings can be changed, the key "glove_embeddings" in ./config.json
must be changed accordingly.

#### Training

In the configuration file ./config/config.json, under the key "captioning", the "attention" key contains all relevant parameters for training 
and hyperparameters to train the captioning model.

Especially, setting "cuda_device" to "cuda:0" will let run the model on your first logical GPU, "cuda:1" on the second,
and so on.

Execute the script ./avatar_models/captioning/preprocessing.py to cache features for training,

It is important, to use the same configuration in ./config/config.json under the key "captioning" for both preprocessing
and training.

The training can be started by running ./avatar_models/captioning/visual_attention_simple.py


#### Evaluation

First download the pretrained model from:


https://drive.google.com/file/d/1ZcCXm9F6T8AbqCGBpDcom4rbDgQyXNr6/view?usp=sharing

Unpack under ./

You should end up with training files under ./pretrained/captioning/ (the most important file being the tokenizer,pickle)
and the saved model files  under ./pretrained/captioning/checkpoints

Run the script `avatar_models/captioning/evaluate.py` to verify that the model can be loaded correctly.


### CATR model

In the configuration file ./config/config.json, under the key "captioning", the "catr" key contains all relevant parameters for training 
and hyperparameters to train the captioning model.

Especially, setting "cuda_device" to "cuda:0" will let run the model on your first logical GPU, "cuda:1" on the second,
and so on.

## VQA

### LXMERT (Huggingface)

If the installations works correctly, all dependencies and weights will be downloaded automatically after the first 
use.

In the configuration file ./config/config.json, under the keys "vqa" / "lxmert", you can change the maximum length of the 
question and switch between "vqa" or "gqa" models. You can also assign directly a GPU with cuda_device, e.g "cuda:0" 
for the first GPU, "cuda:1" for the second GPU or none / "cpu" if no GPUs are available.

To see an example how to run an inference, run models/vqa/lxmert/lxmert_eval.py

The LXMERT models have been also fine tune on our synthetic VQA dataset (under `./data/ade20k_vqa`).

The models are available under: https://drive.google.com/drive/folders/1-xX5ZAEvn6yxfp0EVwLVsODwefOV4PYv

To use the models, create the directories:

`pretrained/vqa/lxmert/vqa`
`pretrained/vqa/lxmert/gqa`

download the gqa model under `pretrained/vqa/lxmert/gqa` 
download the vqa model under `pretrained/vqa/lxmert/vqa`

If you want to use the fine-tuned model on GQA, you need to adjust the configuration keys under "vqa", "lxmert" as 
follow:
"model" : "gqa"
"fine_tuning" : "model_file" : "lxmert_gqa_epoch3.pkl"
"fine_tuning" : "use_pretrained": true

For the fine-tuned model on VQA, change the configuration similarly: 

"model" : "vqa"
"fine_tuning" : "model_file" : "lxmert_vqa_epoch3.pkl"
"fine_tuning" : "use_pretrained": true

Setting "fine_tuning" : "use_pretrained": false will allow to use the models as provided by Huggingface.

### Evaluation on the official VQA metrics

https://drive.google.com/file/d/1EWMHAafdAba2wUv56bvdg8UrV9h9rw6V/view?usp=sharing

Unpack under ./

You should end up with all files under ./avatar_models/vqa/attentions/

We do not need the pretrained model from this archive, only the pre-processed Questions / Answers from the VQA dataset.

These are used in the script./util/utils/vqa_eval.py to perform the VQA evaluation for any model.

Please also unpack all files found under ./data/ade20k_vqa in the directory defined in the ./config/config.json file, 
under the key "ade20k_vqa_dir" (should be the directory `./data/ade20k_vqa`), in order to be able to run the ADE20K evaluation for VQA.

`cd /data/ade20k_vqa`
`tar xvf merged_synthetic_vqa.tar.gz`


## Output Game Statistics from a Slurk Log

If you have extracted a game log from a slurk server (as a reminder, go to the directory where slurk is installed 
and run: `sh scripts/get_logs.sh 00000000-0000-0000-0000-000000000000 avatar_room > your_file_name.log` to extract the logs)

You can run the following to get some game statistics:

`python ./config/score_game.py --file your_file_name.log --dir ../data/game_logs`



# Installation

Important: if you face errors concerning the import of modules, please export this project to you python path:
`export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/sose21-pm-language-and-vision-g1/"`


You can install the scripts to be available from the shell (where the python environment is accessible).

For this simply checkout this repository and perform `python setup.py install` from within the root directory. This will
install the app into the currently activate python environment. Also perform `pip install -r requirements.tx` to install 
additional dependencies. After installation, you can use the `game-setup`
, `game-master` and `game-avatar` cli commands (where the python environment is accessible).

**Installation on the Jarvis Remote Server**

Install for your own user (no sudo)

`pip install virtualenv`

Check out the project from github and execute 
`cd sose21-pm-language-and-vision-g1`

Create a virtual environmen named venv

`virtualenv venv`

Active the created environment:

`source venv/bin/activate`

Install the pytorch matching the remote server CUDA version, in this case:

`pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`

Install the remaining libraries with 

`pip install -r requirements_server.txt`

Install the avatar with:

`pip install .` or `python setup.py install`

To install the latest update from the github repo, just type:

`git pull`
`pip install .` or `python setup.py install`

If git complains about changed files, the following reverts local changes, you should then be able to pull latest changes:
`git checkout .`

You can simply deactivate the environment by typing:

`deactivate`

To check that Tensorflow and Pytorch are installed with GPU support, just run:

`python check_gpu.py`

**Installation for developers on remote machines**
Run `update.sh` to install the project on a machine. This shell script simply pulls the latest changes and performs the
install from above. As a result, the script will install the python project as an egg
into `$HOME/.local/lib/pythonX.Y/site-packages`.

You have to add the install directory to your python path to make the app
available `export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/pythonX.Y/site-packages`

Notice: Use the python version X.Y of your choice. Preferebly add this export also to your `.bashrc`.

# Deployment

## A. Run everything on localhost

### Prepare servers and data

#### 1. Start slurk

Checkout `slurk` and run slurk `local_run`. This will start slurk on `localhost:5000`. This will also create the default
admin token.

#### 2. Download and expose the dataset

Download and unpack the ADE20K dataset. Go into the images training directory and start a http server as the server
serving the images. You can use `python -m http.server 8000` for this.

#### 3. Create the slurk game room and player tokens

Checkout `clp-sose21-pm-vision` and run the `game_setup_cli` script or if installed, the `game-setup` cli. By default,
the script expects slurk to run on `localhost:5000`. If slurk runs on another machine, then you must provide
the `--slurk_host` and `--slurk_port` options. If you do not use the default admin token, then you must use
the `--token` option to provide the admin token for the game setup.

The script will create the game room, task and layout using the default admin token via slurks REST API. This will also
create three tokens: one for the game master, player and avatar. See the console output for these tokens. You can also
manually provide a name for the room.

### Prepare clients and bots

#### 1. Start the game master bot

Run the `game_master_cli --token <master-token>` script or the `game-master --token <master-token>` cli. This will
connect the game master with slurk. By default, this will create image-urls that point to `localhost:8000`.

#### 2. Start a browser for the 'Player'

Run a (private-mode) browser and go to `localhost:5000` and login as `Player` using the `<player-token>`.

#### 3. Start a browser for the 'Avatar' or start the avatar bot

**If you want to try the avatar perspective on your own**, then run a different (private-mode) browser and go
to `localhost:5000` and login as `Avatar` using the `<avatar-token>`.

**If the avatar is supposed to be your bot**, then run the `game_avatar_cli --token <avatar-token>` script or the
the `game-avatar --token <avatar-token>` cli. This will start the bot just like with the game master.

Note: This works best, when the game master is joining the room, before the avatar or the player. The order of player
and avatar should not matter. If a game seems not starting or the avatar seems not responding try to restart a game
session with the `/start` command in the chat window.

Another note: The dialog history will be persistent for the room. If you want to "remove" the dialog history, then you
have to create another room using the `game-setup` cli (or restart slurk and redo everything above). The simple avatar
does not keep track of the dialog history.

## B. Run everything with ngrok (temporarily externally available)

This setup will allow you to play temporarily with others over the internet.

First, do everything as in *Run everything on localhost: Prepare servers and data*.

### Prepare ngrok

1. Externalize the slurk server using `ngrok http 5000` which will give you something like `<slurk-hash>.ngrok.io`
1. Externalize the image server using `ngrok http 8000` which will give you something like `<image-hash>.ngrok.io`

### Prepare clients and bots

#### 1. Start the game master bot

Run the `game_master_cli --token <master-token> --image_server_host <image-hash>.ngrok.io --image_server_port 80`. This
will connect the game master with slurk. This will create image-urls that point to `<image-hash>.ngrok.io:80` which
makes the images accessible over the internet. If you run the game master on the same host as slurk, then the game
master will automatically connect to `localhost:5000`. If you run the game master on another machine than slurk, then
you probably want to also provide the `--slurk_host <slurk-hash>.ngrok.io` and `--slurk_port 80` options.

#### 2. Start a browser for the 'Player'

Run a (private-mode) browser and go to `<slurk-hash>.ngrok.io` and login as `Player` using the player token. You might
have to wait a minute until you can also connect as the second player.

#### 3. Start a browser for the 'Avatar'

Run a different (private-mode) browser and go to `<slurk-hash>.ngrok.io` and login as `Avatar` using the avatar token.
If the avatar is a bot, then the bot will have to use the token, when the connection is about to be established, just
like with the game master.
