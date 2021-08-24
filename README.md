# A MapWorld Avatar Game (SoSe21 PM Vision)

# Repository Structure
This project is the Avatar Game. It has the following structure:

    ├── avatar                              # Game basis
    │   ├── mapworld                        # The map with the images
    │   ├── resources                       # JSON for the layout and the images
    │   ├── scripts                         # To initialize the game
    │   │   ├── __init__.py                 # Load resources
    │   │   ├── game.py                     # Start the game
    │   │   ├── game_avatar.py              # Start the avatar
    │   │   ├── game_avatar_custom.py       # Functional copy of above
    │   │   ├── game_avatar_slurk.py        # Start avatar in slurk
    │   │   ├── game_master_slurk.py        # Start master in slurk
    │   │   └── game_master_standalone.py   # Start master
    ├── avatar_models                       # The VQA and captioning models
    │   ├── captioning                      # The captioning model catr
    │   ├── utils                           # utils and scoring 
    │   │   ├── bleu.py                     # Scores for captioning, BLEU and SPICE
    │   │   ├── create_ade20k_vqa_dataset.py# Prepare the ADE20K for VQA
    │   │   ├── util.py                     # Some helpful functions
    │   │   └── vqa_eval                    # Evaluation of VQA model
    │   ├── vqa                             # VQA LXMERT model
    ├── config                              # Config folder                 
    │   ├── config.json                     # Project Parameters depending on server
    |   ├── dev_config.json                 # Project Parameters for local development
    │   ├── score_game.py                   # Runs game metrics based on Slurk logs
    |   └── utils.py                        # Read the config file
    ├── data                                # ADE20K and game_logs
    ├── notebooks                           # Shows some models output in Jupyter notebooks
    ├── pretrained                          # Directory where to put some of the pretrained model when using them
    ├── rasa                                # Contains the fine tuned RASA model for NLU
    ├── results                             # Text output of the different metrics displayed in the final report
    ├── tests                               # Some MapWorld tests
    └── setup.py                            # To install the game
    

Remark for the configuration (see `config/config.json`):

If some expected directories are missing, leading to some errors, try to execute `config/util.py` (with sudo if necessary)
It will try to create the missing directories needed in some cases.

Also, to make any configuration change effective, you will need to ALWAYS redeploy the whole
project by executing `pip install .`

The size and the characteristics of the map can be changed with following keys in the 
`config/config.json`

- "map_size" : set the size for a square labyrinth. Default is 4.
- "map_ambiguity" : assign a number of adjacent similar rooms. Default is 2
- "map_rooms" : Number of rooms to be populated in the labyrinth. Cannot exceed map_size*map_size, must be a multiple of ambiguity. Default is 8.

Changing the value for similar rooms might increase the number of random crash (problem when sampling from similar room).
In that case the game master and the bot needs to be restarted.


# Pretrained Models

## Captioning

### Attention Model

#### MSCOCO Dataset

In the configuration file ./config.json, the key "ms_coco_dir" indicates the location of the MSCOCO dataset,
which is required to train the captioning model.

Let's call the directory defined for the key "ms_coco_dir" [MS_COCO_DIR]

Download the following files and unzip them under [MS_COCO_DIR]:

- http://images.cocodataset.org/zips/train2017.zip

- http://images.cocodataset.org/zips/val2017.zip

- http://images.cocodataset.org/annotations/annotations_trainval2017.zip

This would result in 3 new directories containing images and annotations:

- [MS_COCO_DIR]/train2017
- [MS_COCO_DIR]/val2017
- [MS_COCO_DIR]/annotations

#### Glove embeddings (optional)

The glove embeddings can be used. For that:
1. Download the files from http://nlp.stanford.edu/data/glove.6B.zip
2. Unzip the content under ./pretrained/gloves
3. The location for the downloaded Glove embeddings can be changed, the key "glove_embeddings" in ./config.json
must be changed accordingly.

#### Training

In the configuration file ./config/config.json, under the key "captioning", the "attention" key contains all relevant parameters for training 
and hyperparameters to train the captioning model.

Especially, setting "cuda_device" to "cuda:0" will let run the model on your first logical GPU, "cuda:1" on the second,
and so on.

1. Execute the script ./avatar_models/captioning/preprocessing.py to cache features for training.

It is important, to use the same configuration in ./config/config.json under the key "captioning" for both preprocessing
and training.
2. The training can be started by running ./avatar_models/captioning/visual_attention_simple.py


#### Evaluation

1. Download the pretrained model from: https://drive.google.com/file/d/1ZcCXm9F6T8AbqCGBpDcom4rbDgQyXNr6/view?usp=sharing
2. Unpack under ./

You should end up with training files under ./pretrained/captioning/ (the most important file being the tokenizer,pickle)
and the saved model files  under ./pretrained/captioning/checkpoints

3. Run the script `avatar_models/captioning/evaluate.py` to verify that the model can be loaded correctly.


### CATR model

In the configuration file ./config/config.json, under the key "captioning", the "catr" key contains all relevant parameters for training 
and hyperparameters to train the captioning model.

Especially, setting "cuda_device" to "cuda:0" will let run the model on your first logical GPU, "cuda:1" on the second,
and so on.

You can run `./avatar_models/captioning/catr/predict.py` to see an example of inference.

### Running BLEU-4 and SPICE score for Attention and CATR model.

To run the captioning evaluation, the additional captions from the Tell Me More fork must be used:

https://github.com/rafiberlin/image-description-sequences

And be configured under the key `ade20k_caption_dir` in the configuration file ./config/config.json accordingly.

The metrics can be run with:

`python ./avatar_models/utils/bleu.py`

## VQA

### Question - Answer Dataset creation for ADE20K

We created Question Answers pairs for ADE20K using 2 methods:

- we used object annotation for ADE20K to create simple yes/no questions with fixed templates
- we re-used a github project ( https://github.com/patil-suraj/question_generation, relying on huggingface transformers)
to create Question Answer pairs based on the localized narrative captions for ADE20K (https://storage.googleapis.com/localized-narratives/annotations/ade20k_train_captions.jsonl)

The resulting dataset is stored under `./data/ade20k_vqa/merged_synthetic_vqa.tar.gz` (must be unzipped).

As many members worked on the dataset creation, we do not have one process to create the
final result in one pass (especially, we didn't use a fixed random seed...). However, all functions available
to create the synthetic dataset are in the repository.



In ./config/config.json, the key "ade20k_dir" contains the path to the directory containing the relevant ADE20K external 
resources such as:
- the original ADE20K images and annotations (in a folder named "images")
- the localized narrative annotations (textual captions only) under folder named "annotations"
- the preprocessed ADE20K annotations saved as pandas data frame export under the folder preprocessed_dfs (as found on the jarvis server under `data/ImageCorpora/ADE20K_2016_07_26/preprocessed_dfs`)

We will refer to the main ADE20K directory under the "ade20k_dir" config key as [ADE20K_DIR].

[ADE20K_DIR] must be prepared to contain the previously listed resources; the configuration file `./config/config.json`
must be amended accordingly.

Once this is done, the next actions should help to recreate the dataset successfully.


The first step is to create the yes/no questions with:

`python ./avatar_models/utils/create_ade20k_vqa_dataset.py`

It creates a `ade20k_vqa.jsonl` file under [ADE20K_DIR].

The second step is to create the Question Answer pairs with transformer architecture:

`python ./avatar_models/vqa/create_qa_pairs.py`

It creates a `ade20k_qa_cleaned.json` file under [ADE20K_DIR].

The `ade20k_qa_cleaned.json` needs to be further processed, to add the actual images paths.
This is done using `add_image_path_qa_data()` in `./avatar_models/utils/util.py`, which will 
output `ade20k_qa_cleaned_with_image_path.json` under [ADE20K_DIR].


Finally, the function `merge_synthetic_qa()` in `./avatar_models/utils/util.py` will merge 
`ade20k_qa_cleaned_with_image_path.json` and `ade20k_vqa.jsonl` into the set of files found under

`./data/ade20k_vqa/merged_synthetic_vqa.tar.gz`.

### LXMERT (Huggingface)
The code for integrating LXMERT in our project comes from :

`https://github.com/huggingface/transformers/blob/master/examples/research_projects/lxmert/demo.ipynb`

If the installations works correctly, all dependencies and weights will be downloaded automatically after the first 
use.

In the configuration file `./config/config.json`, under the keys "vqa" / "lxmert", you can change the maximum length of the 
question and switch between "vqa" or "gqa" models. You can also assign directly a GPU with cuda_device, e.g "cuda:0" 
for the first GPU, "cuda:1" for the second GPU or none / "cpu" if no GPUs are available.

#### Finetuning LXMERT

1. run `python compute_features.py` — this will compute FRCNN features for the ADE20K Dataset and save them into a folder to ensure faster training.
2. run `pyhon finetune.py` — this will fine-tune the specified model for 3 epochs. The type of the model (vqa or gqa), type of device (cuda or cpu) and other stuff are specified in the  config file
3. the script expects the dataset ("merged_synthetic_vqa_splits.json") to be in this directory, so add it here or change the path in the script accordingly


To see an example how to run an inference, run `models/vqa/lxmert/lxmert_eval.py`

The LXMERT models have been also fine tuned on our synthetic VQA dataset (under `./data/ade20k_vqa`).

The models are available under: https://drive.google.com/drive/folders/1-xX5ZAEvn6yxfp0EVwLVsODwefOV4PYv

To use the models, create the directory:

`./pretrained/vqa/lxmert/`


Download the vqa model under `pretrained/vqa/lxmert/` from https://drive.google.com/file/d/1SCqFSLRaIwZD5AdTTkWKs6qhvHfjrIy9/view?usp=sharing, and 
unpack the content with `tar xvf vqa.tar.gz`, you should end up with  `./pretrained/vqa/lxmert/vqa/epoch1.pkl`.

Download the gqa model under `pretrained/vqa/lxmert/` from https://drive.google.com/file/d/1FHXernav2TNDl8txss4ATV0gPuaUQSEe/view?usp=sharing
unpack the content with `tar xvf gqa.tar.gz`, you should end up with  `./pretrained/vqa/lxmert/gqa/epoch1.pkl`


If you want to use the fine-tuned model on GQA, you need to adjust the configuration keys under "vqa", "lxmert" as 
follow:
"model" : "gqa"
"fine_tuning" : "model_file" : "epoch1.pkl"
"fine_tuning" : "use_pretrained": true

For the fine-tuned model on VQA, change the configuration similarly: 

"model" : "vqa"
"fine_tuning" : "model_file" : "epoch1.pkl"
"fine_tuning" : "use_pretrained": true

Setting "fine_tuning" : "use_pretrained": false will allow to use the models as provided by Huggingface.


Note: these a fine tuned model for 3 epochs but their performance on the VQA 2.0 task is much worse...

VQA Fine tuned 3 epochs: https://drive.google.com/file/d/1KWApviWWF3qCeXhMgSzPpPHd8JUwZEQN/view?usp=sharing

GQA Fine tuned 3 epochs:  https://drive.google.com/file/d/1r15rAqsHxwbhyvwlZkI1dZ7T1LsTlUlL/view?usp=sharing

### Evaluation on the official VQA metrics

1. Download `attention_vqa_val_acc_0.394.tar.gz` from https://drive.google.com/file/d/1EWMHAafdAba2wUv56bvdg8UrV9h9rw6V/view?usp=sharing

2. Unpack under ./

You should end up with all files under ./avatar_models/vqa/attentions/

We do not need the pretrained model from this archive, only the pre-processed Questions / Answers from the VQA dataset.

These are used in the script./util/utils/vqa_eval.py to perform the VQA evaluation for any model.

3. Please also unpack all files found under `./data/ade20k_vqa` in the directory defined in the `./config/config.json file`, 
under the key "ade20k_vqa_dir" (should be the directory `./data/ade20k_vqa`), in order to be able to run the ADE20K evaluation for VQA.

    `cd /data/ade20k_vqa`

    `tar xvf merged_synthetic_vqa.tar.gz`


4. The evaluation of the different models can be performed with:

`python avatar_models/utils/vqa_eval.py`

## RASA

In the configuration file `./config/config.json`, the key "rasa" indicates the location of the finetuned NLU model and
necessary files.

To train the NLU model, type inside the rasa directory: 

`rasa train`

This trains a model using the NLU data and stories and saves it in `./models`. To then access the nlu and core functions locally without running
the HTTP API, decompress the model file (ending with tar.gz) to reveal the folders `./models/core` and `./models/nlu`.

For finetuning, the following files have to be edited for the changes to take effect:
`./domain.yml`
`./data/nlu.yml`
`./data/stories.yml`
`./data/rules.yml`

After adjusting the files, repeat training and (if no errors occur) decompressing the model file. 

## Output Game Statistics from a Slurk Log

If you have extracted a game log from a slurk server (as a reminder, go to the directory where slurk is installed 
and run: `sh scripts/get_logs.sh 00000000-0000-0000-0000-000000000000 avatar_room > your_file_name.log` to extract the logs)

You can run the following to get some game statistics:

`python ./config/score_game.py --file your_file_name.log --dir ../data/game_logs`


# Installation

Important: if you face errors concerning the import of modules, please export this project to you python path:
`export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/sose21-pm-language-and-vision-g1/"`


You can install the scripts to be available from the shell (where the python environment is accessible).

For this simply checkout this repository and perform `python install .` from within the root directory. This will
install the app into the currently activate python environment. Also perform `pip install -r requirements.tx` to install 
additional dependencies. After installation, you can use the `game-setup`
, `game-master` and `game-avatar` cli commands (where the python environment is accessible).

**Installation on the Jarvis Remote Server**

1. Install virtualenv for your own user (no sudo)
   
`pip install virtualenv`

2. Check out the project from github and execute 

`cd sose21-pm-language-and-vision-g1`

3. Create a virtual environmen named venv

`virtualenv venv`

4. Active the created environment:

`source venv/bin/activate`

5. Install the pytorch matching the remote server CUDA version, in this case:

`pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`

6. Install the remaining libraries with 

`pip install -r requirements_server.txt`

7. Install the avatar with:

`pip install .` or `python setup.py install`

8. To install the latest update from the github repo, just type:

`git pull`
`pip install .` or `python setup.py install`

10. You can simply deactivate the environment by typing:

`deactivate`

11. To check that Tensorflow and Pytorch are installed with GPU support, just run:

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
