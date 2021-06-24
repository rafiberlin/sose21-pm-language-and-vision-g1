import torch
import torch.nn.functional

from transformers import BertTokenizer
from PIL import Image

from avatar_models.captioning.catr.datasets import coco, utils
from avatar_models.captioning.catr.configuration import Config
from config.util import get_config
import numpy as np
import os

"""
parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, help='path to image', required=True)
parser.add_argument('--v', type=str, help='version', default='v3')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
args = parser.parse_args()
#image_path = args.path
version = args.v
checkpoint_path = args.checkpoint


if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
else:
    print("Checking for checkpoint.")
    if checkpoint_path is None:
      raise NotImplementedError('No model to chose from!')
    else:
      if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
      print("Found checkpoint! Loading!")
      model,_ = caption.build_model(config)
      print("Loading Checkpoint...")
      checkpoint = torch.load(checkpoint_path, map_location='cpu')
      model.load_state_dict(checkpoint['model'])
"""

class CATRInference():
    def __init__(self):
        self.config = Config()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = self.config.max_position_embeddings
        self.start_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer._cls_token)
        self.end_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer._sep_token)
        self.model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
        catr_config = get_config()["captioning"]["catr"]
        self.cuda_device = catr_config["cuda_device"]
        self.beam_size = catr_config["beam_size"]

        if type(self.cuda_device) is str and self.cuda_device.startswith("cuda"):
            print("Use CATR Model with GPU", self.cuda_device)
            self.model.cuda(self.cuda_device)
        else:
            print("Use CATR Model with CPU")

    def create_caption_and_mask(self):



        self.caption_template = torch.zeros((1, self.max_length), dtype=torch.long)
        self.mask_template = torch.ones((1, self.max_length), dtype=torch.bool)

        self.caption_template[:, 0] = self.start_token
        self.mask_template[:, 0] = False

        return self.caption_template, self.mask_template
    @torch.no_grad()
    def infer_beam(self, image_path):

        image = Image.open(image_path)
        image = coco.val_transform(image)
        image = image.unsqueeze(0)
        beam_size = self.beam_size
        caption, cap_mask = self.create_caption_and_mask()
        previous_log_prob = torch.zeros((beam_size, 1))

        if self.cuda_device.startswith("cuda"):
            image = image.cuda(self.cuda_device)
            caption = caption.cuda(self.cuda_device)
            cap_mask = cap_mask.cuda(self.cuda_device)
            previous_log_prob = previous_log_prob.cuda(self.cuda_device)
        # model.eval()

        predictions = self.model(image, caption, cap_mask)
        predictions = torch.nn.functional.log_softmax(predictions[:, 0, :])
        previous_log_prob, candidate_indices = torch.topk(predictions, beam_size)
        preds = {i: np.zeros(self.max_length, dtype=int) for i in range(beam_size)}
        for i in range(beam_size):
            preds[i][0] = candidate_indices[0][i]
        # Copy entries a number of time equal to the beam size (the number of alternative paths)
        # 1 means the dimensions stay untouched
        image = image.repeat(beam_size, 1, 1, 1)
        caption = caption.repeat(beam_size, 1)
        cap_mask = cap_mask.repeat(beam_size, 1)
        candidates = []
        caption[:, 0] = candidate_indices
        cap_mask[:, 0] = False
        for step in range(1, self.max_length - 1):
            predictions = self.model(image, caption, cap_mask)
            predictions = torch.nn.functional.log_softmax(predictions[:, step, :])
            candidates_log_prob, candidate_indices = torch.topk(predictions, beam_size)
            candidates_log_prob = torch.reshape(candidates_log_prob + previous_log_prob, (-1,))
            candidate_indices = torch.reshape(candidate_indices, (-1,))
            current_top_candidates, current_top_candidates_idx = torch.topk(candidates_log_prob, k=beam_size)

            # Do the mapping best candidate and "source" of the best candidates
            k_idx = torch.gather(candidate_indices, dim=0, index=current_top_candidates_idx)
            prev_idx = torch.floor(current_top_candidates_idx / beam_size).to(torch.int32)

            previous_log_prob = torch.unsqueeze(current_top_candidates, dim=1)
            np_prev_idx = prev_idx.cpu().numpy()
            # Overwrite the previous predictions due to the new best candidates
            temp = caption.clone()
            for i in range(prev_idx.shape[0]):
                temp[i][:step] = caption[np_prev_idx[i]][:step]
            caption = temp
            preds = {i: preds[np_prev_idx[i]].copy() for i in range(prev_idx.shape[0])}

            stop_idx = []
            for i in range(k_idx.shape[0]):
                preds[i][step] = k_idx[i]
                if k_idx[i] == self.end_token:
                    stop_idx.append(i)

            # remove all finished captions and adjust all tensors accordingly...
            if len(stop_idx):
                for i in reversed(sorted(stop_idx)):
                    candidate = preds.pop(i)
                    loss = current_top_candidates[i]
                    length = np.where(candidate == self.end_token)[0] + 1
                    normalized_loss = loss / float(length)
                    candidates.append((candidate, normalized_loss))
                beam_size = beam_size - len(stop_idx)
                if beam_size > 0:
                    left_idx = torch.LongTensor([i for i in range(k_idx.shape[0]) if i not in stop_idx])
                    k_idx = torch.LongTensor([k_idx[i] for i in range(k_idx.shape[0]) if i not in stop_idx])
                    if self.cuda_device.startswith("cuda"):
                        left_idx = left_idx.cuda(self.cuda_device)
                        k_idx = k_idx.cuda(self.cuda_device)
                    # current_top_candidates = torch.IntTensor(
                    #     [current_top_candidates[i] for i in range(current_top_candidates.shape[0]) if
                    #      i not in stop_idx])
                    caption = torch.index_select(caption, dim=0, index=left_idx)
                    cap_mask = torch.index_select(cap_mask, dim=0, index=left_idx)
                    # now that the finished sentences have been removed, we need to update the predictions dict accordingly
                    for i, key in enumerate(sorted(preds.keys())):
                        preds[i] = preds.pop(key)
                else:
                    break  # No sequences unfinished

            # predicted_id = torch.argmax(predictions, axis=-1)
            pass
            # if predicted_id[0] == self.end_token:
            #     #return caption
            #     break
            caption[:, step] = k_idx
            cap_mask[:, step] = False

        output = self.tokenizer.decode(caption[0].tolist(), skip_special_tokens=True)
        return output

    @torch.no_grad()
    def infer(self, image_path):

        image = Image.open(image_path)
        image = coco.val_transform(image)
        image = image.unsqueeze(0)

        caption, cap_mask = self.create_caption_and_mask()
        if self.cuda_device.startswith("cuda"):
            image = image.cuda(self.cuda_device)
            caption = caption.cuda(self.cuda_device)
            cap_mask = cap_mask.cuda(self.cuda_device)
        #model.eval()
        for i in range(self.config.max_position_embeddings - 1):
            predictions = self.model(image, caption, cap_mask)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)

            if predicted_id[0] == 102:
                #return caption
                break
            caption[:, i+1] = predicted_id[0]
            cap_mask[:, i+1] = False
        output = self.tokenizer.decode(caption[0].tolist(), skip_special_tokens=True)
        return output


if __name__ == "__main__":
    image_path = "/home/rafi/_datasets/ADE20K/images/training/u/utility_room/ADE_train_00019432.jpg"
    catr = CATRInference()
    output = catr.infer_beam(image_path)
    #result = catr.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    #result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output)
