import torch

from transformers import BertTokenizer
from PIL import Image

from avatar_models.captioning.catr.datasets import coco, utils
from avatar_models.captioning.catr.configuration import Config
from config.util import get_config
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
        self.cuda_device = get_config()["captioning"]["catr"]["cuda_device"]
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
                return caption

            caption[:, i+1] = predicted_id[0]
            cap_mask[:, i+1] = False

        return caption


if __name__ == "__main__":
    image_path = "/home/rafi/_datasets/ADE20K/images/training/u/utility_room/ADE_train_00019432.jpg"
    catr = CATRInference()
    output = catr.infer(image_path)
    result = catr.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    #result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(result.capitalize())
