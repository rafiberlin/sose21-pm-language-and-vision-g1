import pickle
import os
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
from transformers import AdamW # is this needed for training?
from config.util import get_config
from avatar_models.vqa.lxmert.utils import get_data
    
class ADE20K_Dataset(Dataset):
    def __init__(self, model_type, image_paths, questions, answers, question_length, features_folder="precomputed_features/training"):
        self.image_paths = image_paths
        self.questions = questions
        self.answers = answers
        self.question_length = question_length
        self.vocab, self.label2ans, self.ans2label, self.tokenizer = self.load_vocab(model_type)
        # filter out OOV rows
        good_indices = [index for index, ans in enumerate(self.answers) if ans in self.vocab]
        self.image_paths = np.array(self.image_paths)[good_indices]
        self.questions = np.array(self.questions)[good_indices]
        self.answers = np.array(self.answers)[good_indices]
        self.features_folder = features_folder
    def load_vocab(self, model_type):
        if model_type == "vqa":
            tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-vqa-uncased")
            VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"
            answers = get_data(VQA_URL)
            label_2_ans = {key:value for key, value in enumerate(answers)}
            ans_2_label = {value:key for key, value in label_2_ans.items()}
        else:
            tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-gqa-uncased")
            GQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/gqa/trainval_label2ans.json"
            answers = get_data(GQA_URL)
            label_2_ans = {key:value for key, value in enumerate(answers)}
            ans_2_label = {value:key for key, value in label_2_ans.items()}
        return (answers, label_2_ans, ans_2_label, tokenizer)
    
    def load_features(self, img_path):
        "Loads precomputed FRCNN features for the image from the precomputed_features folder."
        # Add the .pickle extension
        img_path += ".pickle"
        output_dict = pickle.load(open(os.path.join(self.features_folder, img_path), "rb"))
        return output_dict
  
    def __getitem__(self, idx):
        img_path = self.image_paths[idx].split("/")[-1] # "training/b/bedroom/ADE_train_00003661.jpg" --> "ADE_train_00003661.jpg"
    
        output_dict = self.load_features(img_path)
        features = output_dict["roi_features"]    
        normalized_boxes = output_dict["normalized_boxes"]
        
        q = self.questions[idx]
        input = self.tokenizer(
            q,
            padding="max_length", 
            max_length=self.question_length,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
      
        answer = self.ans2label[self.answers[idx]]

        return (input, features, normalized_boxes), answer
  
    def __len__(self):
        return len(self.answers)

def train(model_type, device, train_dataset, test_dataset, batch_size, n_epochs=3, learning_rate=5e-5):
    if model_type == "vqa":
        model_checkpoint = "unc-nlp/lxmert-vqa-uncased"
        print("Loading {}".format(model_checkpoint))
        model = LxmertForQuestionAnswering.from_pretrained(model_checkpoint).to(device)

    elif model_type == "gqa":
        model_checkpoint = "unc-nlp/lxmert-gqa-uncased"
        print("Loading {}".format(model_checkpoint))
        model = LxmertForQuestionAnswering.from_pretrained(model_checkpoint).to(device)
    else:
        raise ValueError(f"Wrong model type{model_type}, choose gqa or vqa")

    save_dir = os.path.join(MODEL_DIR, model_type)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    
    model.train()

    optim = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    total_loss = 0
    print("Starting training.")
    for epoch in range(n_epochs):
        print("Epoch {}".format(epoch+1))
        for batch in tqdm(train_loader):
            optim.zero_grad()
  
            input, features, normalized_boxes = batch[0]
            input_ids = input["input_ids"].to(device)
            attention_mask = input['attention_mask'].to(device)
            token_type_ids = input['token_type_ids'].to(device)
            
            labels = batch[1].to(device)
            
            # forward pass
            outputs = model(input_ids=input_ids.squeeze(1), 
                            attention_mask=attention_mask.squeeze(1),
                            visual_feats=features.squeeze(1).to(device),
                            visual_pos=normalized_boxes.squeeze(1).to(device),
                            token_type_ids=token_type_ids.squeeze(1),
                            output_attentions=False,
                            labels=labels)
            # calculate loss and execute a training step
            loss = outputs.loss
            loss.backward()
            total_loss += loss
            optim.step()
            
        print('Average loss: {:.4f}'.format(total_loss / len(train_loader)))
        total_loss = 0
        
        # run evaluation after each epoch
        evaluate(model=model, device=device, test_dataset=test_dataset, batch_size=batch_size)
        
        # Save checkpoint
        state = {
             'state_dict': model.state_dict(),
             'optimizer': optim.state_dict()
        }
        print("Saving a model after epoch ".format(epoch+1))
        torch.save(
            state,
            os.path.join(save_dir,
                         'epoch' + str(epoch+1) + '.pkl')
        )
        print("Saved!")

def evaluate(model, device, test_dataset, batch_size):
    model.eval()
    
    eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    total = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            labels = batch[1].to(device)

            input, features, normalized_boxes = batch[0]
            input_ids = input["input_ids"]
            attention_mask = input['attention_mask']
            token_type_ids = input['token_type_ids']

            # forward pass
            outputs = model(input_ids=input_ids.squeeze(1).to(device), 
                            attention_mask=attention_mask.squeeze(1).to(device), 
                            visual_feats=features.squeeze(1).to(device),
                            visual_pos=normalized_boxes.squeeze(1).to(device),
                            token_type_ids=token_type_ids.squeeze(1).to(device),
                            output_attentions=False)
            total += sum(outputs['question_answering_score'].argmax(-1) == labels)
        print('Accuracy: {:.2f}%'.format(float(total * 100 / len(test_dataset))))
        

if __name__ == "__main__":
    # Get configuration
    config = get_config()
    BATCH_SIZE = config['vqa']['lxmert']['fine_tuning']["batch_size"]
    EPOCH = config['vqa']['lxmert']['fine_tuning']["epoch"]
    MODEL_TYPE = config['vqa']['lxmert']['model']
    LR = config['vqa']['lxmert']['fine_tuning']["learning_rate"]
    QUESTION_LENGTH = config['vqa']['lxmert']['question_length']
    DEVICE = config['vqa']['lxmert']['cuda_device']
    MODEL_DIR = os.path.join( config["ade20k_vqa_dir"], config['vqa']['lxmert']['fine_tuning']["pretrained_dir"])
    ADE20K_SYNTHETIC_DATASET = os.path.join(config["ade20k_vqa_dir"], config["ade20k_vqa_file_train_test"])
    features_folder = os.path.join(config["ade20k_vqa_dir"], "precomputed_features/training")

    print("DEVICE:", DEVICE)

    # # Create folder for saving the model => moved to get_config()
    # if not os.path.exists("finetuned-lxmert/{}".format(MODEL_TYPE)):
    #     os.makedirs("finetuned-lxmert/{}".format(MODEL_TYPE))
        
    # Load dataset
    print(f"Loading the dataset under {ADE20K_SYNTHETIC_DATASET}")
    with open(ADE20K_SYNTHETIC_DATASET, "r") as f:
        vqa_dataset = json.load(f)
    
    training = pd.DataFrame(vqa_dataset["training"])
    testing = pd.DataFrame(vqa_dataset["testing"])
    
    train_image_paths, train_questions, train_answers = training['image_path'].values, training['question'].values, training['answer'].values
    test_image_paths, test_questions, test_answers = testing['image_path'].values, testing['question'].values, testing['answer'].values
    
    train_dataset = ADE20K_Dataset(MODEL_TYPE, train_image_paths, train_questions, train_answers, QUESTION_LENGTH, features_folder)
    test_dataset = ADE20K_Dataset(MODEL_TYPE, test_image_paths, test_questions, test_answers, QUESTION_LENGTH, features_folder)

    # Fine-tune the model
    train(MODEL_TYPE, DEVICE, train_dataset, test_dataset, BATCH_SIZE, n_epochs=EPOCH, learning_rate=LR)
