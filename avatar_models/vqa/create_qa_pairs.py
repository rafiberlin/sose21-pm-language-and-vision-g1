# you need to clone https://github.com/patil-suraj/question_generation in the directory of the script


import json
import nltk
from question_generation.pipelines import pipeline
from avatar_models.utils.util import get_config
nltk.download('punkt')

def preprocess_qa(qa):
    '''This monstrosity of a function tries to clean the super messy QA pairs.'''
    q = qa['question']
    a = qa['answer']

    q = q.replace("inside of a room", "")
    q = q.replace("of the image", "")
    q = q.replace("in this image", "")
    q = q.replace("in the image", "")
    q = q.replace("the picture of", "")
    q = q.replace("the image taken", "this")
    q = q.replace("image of a", "")
    q = q.replace("of this picture,", "")
    q = q.replace("of this picture", "")
    q = q.replace("of the picture", "")
    q = q.replace("in this picture", "")
    q = q.replace("What are there", "What is")
    q = q.replace("What is the background", "What is in the background")
    q = q.replace("a picture taken", "")
    q = q.replace("taken", "")
    q = q.replace("What is a picture of?", "What do you see?")
    if q.endswith("I can see what?") or q.endswith(", I can see what?"):
        q = q.replace(", I can see what", "")
        q = q.replace(" I can see what", "")
        q = q.lower()
        q = "What can I see " + q
    if q.endswith(", I can see what?") or q.endswith(", we can see what?") or q.endswith(" we can see what?"):
        q = q.replace(", I can see what", "")
        q = q.replace(", we can see what", "")
        q = q.replace(" we can see what", "")
        q = q.lower()
        q = "What is " + q
    clicked = ["Where is the picture clicked?","Where is the image clicked?","Where is the image clicked on?","Where is this picture clicked?","Where does the picture appear to be clicked?","What is the image clicked in?"]
    if q in clicked:
        q = "Is this indoors or outdoors?"
    q = q.replace("In front , we see what?", "What is in the front?")

    q = q.replace("  ", " ")

    a = a.lower()
    a = a.replace("1", "one")
    a = a.replace("2", "two")
    a = a.replace("3", "three")
    a = a.replace("4", "four")
    if a == "outdoors" or a == "indoors":
        q = "Is this outdoors on indoors?"
    if a  in ["red", "blue", "green", "yellow", "white", "black", "purple", "orange", "grey"]:
        q = q.replace("What kind of", "What color")
        q = q.replace("What type of", "What color")
        q = q.replace("What is the name of", "What is the color of")
        q = q.replace("What is", "What color is")
        q = q.replace("type", "color")

    return q,a

def get_multiple_answers(answer):
    '''Splits an answer into multiple answers for questions such as the following:
    question: What can we see in this picture?
    answer:   benches, wooden objects, arches and glass windows
    Because such an answer will be out of vocabulary for the VQA system, the question/answer pair is split into multiple pairs, e.g.:
    (question, benches)
    (question, wooden objects)
    (question, arches)
    (question, glass windows)

    '''
    answer = answer.replace("and", ",")
    answers = answer.split(",")
    answers = [a.strip() for a in answers]
    return answers

# a list of weird questions that are often repeated — these are discarded from the final qa set
shit_list=["What is the name of the building in front of the building?",
          "What can we see ?",
          "What types of chairs are there?",
          "What are the three people ?",
          "What is in the background ?",
          "What types of trees are on the roller coaster?",
          "What are railings on the left side ?",
          "What are chairs, cupboards, lights and bottles?",
          "What are trees, light poles, buildings, vehicles on the road?"]

conf = get_config()
LOCALIZED_NARRATIVES_FILE = conf["ade20k_localized_narratives_train_file"]

# create the pipeline
nlp = pipeline("question-generation")

# open ade20k localized narratives train captions
file = open(LOCALIZED_NARRATIVES_FILE, "r").readlines()
print("Successfully loaded the ADE20K localized narratives training data set.")

# create qa pairs
qa = {}
for ann in annotations:
    d = []
    try:
        qa_pairs = nlp(ann['caption'])
        for pair in qa_pairs:
            dict_ = {}
            dict_['question'] = pair['question']
            dict_['answer'] = pair['answer']
            d.append(dict_)
        qa[ann['image_id']] = d
    except:
        pass

# preprocess/clean the qa pairs
for image in qa:
    for qa_pair in qa[image]:
        q,a = preprocess_qa(qa_pair)
        qa_pair['question'] = q
        qa_pair['answer'] = a

cleaned_qa = {}
for image in qa:
    cleaned_qa[image] = []
    for qa_pair in qa[image]:
        question = qa_pair['question']
        answer = qa_pair['answer']

        if "," in answer: # answers with multiple choices
            answers = get_multiple_answers(answer)
            for ans in answers:
                cleaned_qa[image].append({"question":question, "answer":ans})

        elif answer not in question:
            cleaned_qa[image].append(qa_pair)

# save to disk
with open('ade20k_qa_cleaned.json', 'w') as fp:
    json.dump(nothing, fp)
print("Saved the dataset to disk.")
