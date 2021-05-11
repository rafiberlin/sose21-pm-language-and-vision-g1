import pandas as pd
import os
import tensorflow as tf
import json
from sklearn.preprocessing import LabelBinarizer
import heapq
import pickle
from sklearn.model_selection import train_test_split

contractions = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because",
    "could've": "could have", "couldn't": "could not",
    "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "hadn't": "had not", "hadn't've": "had not have",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
    "he's": "he is", "how'd": "how did",
    "how'll": "how will", "how's": "how is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would",
    "it'll": "it will", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
    "might've": "might have", "mightn't": "might not",
    "must've": "must have", "mustn't": "must not", "needn't": "need not", "oughtn't": "ought not",
    "shan't": "shall not", "sha'n't": "shall not", "she'd": "she would",
    "she'll": "she will", "she's": "she is", "should've": "should have", "shouldn't": "should not",
    "that'd": "that would", "that's": "that is", "there'd": "there had",
    "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would",
    "we'll": "we will", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
    "what're": "what are", "what's": "what is",
    "what've": "what have", "where'd": "where did", "where's": "where is", "who'll": "who will", "who's": "who is",
    "won't": "will not", "wouldn't": "would not",
    "you'd": "you would", "you'll": "you will", "you're": "you are"
}


def getPeopleAnswer(answers):
    answers_dict = {}
    score_dict = {'yes': 3, 'maybe': 2, 'no': 1}
    for _answer in answers:
        score = score_dict[_answer['answer_confidence']]
        if answers_dict.get(_answer['answer'], -1) != -1:
            answers_dict[_answer['answer']] += score
        else:
            answers_dict[_answer['answer']] = score

    return sorted(list(answers_dict.items()), key=lambda x: x[1], reverse=True)[0][0]

def preprocess_english(text):
    '''Given a text this function removes the punctuations and returns the remaining text string'''
    new_text = ""
    text = text.lower()
    i = 0
    for word in text.split():
        if i == 0:
            new_text = contractions.get(word, word)
        else:
            new_text = new_text + " " + contractions.get(word, word)
        i += 1
    return new_text.replace("'s", '')

def preprocess_english_add_tokens(text):
    '''Given a text this function removes the punctuations and returns the remaining text string'''
    new_text = "<start>"
    text = text.lower()
    for word in text.split():
        new_text = new_text + " " + contractions.get(word, word)
    new_text = new_text + " <end>"
    return new_text.replace("'s", '')


if __name__ == "__main__":
    # CReates training and validations sets as pandas frameworks (saved as csv under /checkpoints)
    # label encoder (which encodes as category the best 1000 questions) and the tokenizer and all processed questions as pickle data

    # Beware : VQA uses pictures from MS COCO 2014 => some pictures disapeared in MS COCO 2017...
    VQA_ANNOTATIONS_DIR = "/home/rafi/_datasets/VQA/"
    MS_COCO_DIR = '/home/rafi/_datasets/MSCOCO/'

    annotation_file = os.path.join(VQA_ANNOTATIONS_DIR, "v2_mscoco_train2014_annotations.json")
    question_file = os.path.join(VQA_ANNOTATIONS_DIR, "v2_OpenEnded_mscoco_train2014_questions.json")

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)["annotations"]

    with open(question_file, 'r') as f:
        questions = json.load(f)["questions"]

    # Get the first image
    image_id = annotations[0]["image_id"]
    coco_train = os.path.join(MS_COCO_DIR, "train2017")
    image_path = os.path.join(coco_train, '%012d.jpg' % (image_id))

    questions_df = pd.DataFrame(questions)
    annotations_df = pd.DataFrame(annotations)
    data = pd.merge(questions_df, annotations_df, how='inner', left_on=['image_id', 'question_id'],
                    right_on=['image_id', 'question_id'])


    # I have to check that but I think each line in questions matches the corresponding lines in annotations (answers)

    # img = Image.open(image_path)
    # img.show()

    # Lets try to make this implementation work: https://medium.com/@harshareddykancharla/visual-question-answering-with-hierarchical-question-image-co-attention-c5836684a180
    # https://github.com/harsha977/Visual-Question-Answering-With-Hierarchical-Question-Image-Co-Attention
    # Official API for retrieval: https://github.com/GT-Vision-Lab/VQA/blob/master/PythonHelperTools/vqaDemo.py




    data['derived_answer'] = data["answers"].apply(lambda x: getPeopleAnswer(x))
    data.to_csv(os.path.join(VQA_ANNOTATIONS_DIR, 'data.csv'))






    data['multiple_choice_answer'] = data['multiple_choice_answer'].apply(lambda x: preprocess_english(x))

    X_train, X_val = train_test_split(data, test_size=0.2, random_state=42)

    all_classes = X_train['multiple_choice_answer'].values
    class_frequency = {}

    for _cls in all_classes:
        if (class_frequency.get(_cls, -1) > 0):
            class_frequency[_cls] += 1
        else:
            class_frequency[_cls] = 1

    common_tags = heapq.nlargest(1000, class_frequency, key=class_frequency.get)
    X_train['multiple_choice_answer'] = X_train['multiple_choice_answer'].apply(lambda x: x if x in common_tags else '')

    # removing question which has empty tags
    X_train = X_train[X_train['multiple_choice_answer'].apply(lambda x: len(x) > 0)]

    label_encoder = LabelBinarizer()
    answer_vector_train = label_encoder.fit_transform(X_train['multiple_choice_answer'].apply(lambda x: x).values)
    answer_vector_val = label_encoder.transform(X_val['multiple_choice_answer'].apply(lambda x: x).values)

    ans_vocab = {l: i for i, l in enumerate(label_encoder.classes_)}

    print("Number of clasess: ", len(ans_vocab))
    print("Shape of Answer Vectors in Train Data: ", answer_vector_train.shape)
    print("Shape of Answer Vectors in Validation Data: ", answer_vector_val.shape)




    X_train['question'] = X_train['question'].apply(lambda x: preprocess_english_add_tokens(x))
    X_val['question'] = X_val['question'].apply(lambda x: preprocess_english_add_tokens(x))

    # tokenization
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(X_train['question'].values)
    train_question_seqs = tokenizer.texts_to_sequences(X_train['question'].values)
    val_question_seqs = tokenizer.texts_to_sequences(X_val['question'].values)



    # Padding
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    print("Number of words in tokenizer:", len(tokenizer.word_index))
    ques_vocab = tokenizer.word_index
    question_vector_train = tf.keras.preprocessing.sequence.pad_sequences(train_question_seqs, padding='post')
    question_vector_val = tf.keras.preprocessing.sequence.pad_sequences(val_question_seqs, padding='post',
                                                                        maxlen=question_vector_train.shape[1])

    print("Shape of Question Vectors in Train Data: ", question_vector_train.shape)
    print("Shape of Question Vectors in Validation Data: ", question_vector_val.shape)

    serialized_tokenizer = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "checkpoints/tokenizer.pickle")

    with open(serialized_tokenizer, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X_train.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints/X_train.csv"))
    X_val.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints/X_val.csv"))

    label_encoder_serialized = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "checkpoints/label_encoder.pickle")

    with open(label_encoder_serialized, 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

    serialized_question_vector_train = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "checkpoints/question_vector_train.pickle")

    with open(serialized_question_vector_train, 'wb') as handle:
        pickle.dump(question_vector_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    serialized_question_vector_val = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "checkpoints/question_vector_val.pickle")

    with open(serialized_question_vector_val, 'wb') as handle:
        pickle.dump(question_vector_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

