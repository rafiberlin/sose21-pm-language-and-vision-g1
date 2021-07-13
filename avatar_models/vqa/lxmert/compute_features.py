import pickle
import os 
from tqdm import tqdm
from avatar_models.vqa.lxmert.processing_image import Preprocess
from avatar_models.vqa.lxmert.modeling_frcnn import GeneralizedRCNN
from avatar_models.vqa.lxmert.utils import Config
import torch
from config.util import get_config

def compute_features():

    project_conf = get_config()
    ADE20K_DIR = project_conf["ade20k_dir"]
    train_dir = os.path.join(ADE20K_DIR, "images/training")

    train_save_path = os.path.join(project_conf["ade20k_vqa_dir"], "precomputed_features/training")

    if not os.path.exists(train_dir):
        raise(f"Training Dir {train_dir} does not exist")

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = "cpu"
    tmp_device = project_conf["vqa"]["lxmert"]["cuda_device"]
    if torch.cuda.is_available() and tmp_device is not None and tmp_device.startswith("cuda"):
        print("Enabling CUDA for LXMERT Feature extraction", device)
        device = tmp_device

    checkpoint = "unc-nlp/frcnn-vg-finetuned"
    frcnn_cfg = Config.from_pretrained(checkpoint)
    frcnn_cfg.MODEL.device = device
    print("Loading FRCNN...")
    frcnn = GeneralizedRCNN.from_pretrained(checkpoint, config=frcnn_cfg)
    image_preprocess = Preprocess(frcnn_cfg)
    
    print("Computing features from training images...")

    for root, dirs, files in tqdm(os.walk(train_dir)): #edit path
        for file in files:
            if file.endswith("jpg") and file[0] != ".": # We only care about normal images
                img_file = os.path.join(root, file)
                images, sizes, scales_yx = image_preprocess(img_file)
                output_dict = frcnn(images, sizes, scales_yx=scales_yx, padding="max_detections", max_detections=frcnn_cfg.max_detections, return_tensors="pt")
        
                # Save output_dict to disk
                file_without_extension = file.replace(".jpg", "")
                save_path = os.path.join(train_save_path, file)
                with open(save_path + ".pickle", "wb") as f:
                    pickle.dump(output_dict, f)
                    
#     print("Computing features from validation images...")
#     valid_save_path = "precomputed_features/validation"
#     if not os.path.exists(valid_save_path):
#         os.makedirs(valid_save_path)
        
#     for root, dirs, files in tqdm(os.walk("/data/ImageCorpora/ADE20K_2016_07_26/images/validation")): #edit path
#         for file in files:
#             if file.endswith("jpg") and file[0] != ".": # We only care about normal images
#                 img_file = os.path.join(root, file)
#                 images, sizes, scales_yx = image_preprocess(img_file)
#                 output_dict = frcnn(images, sizes, scales_yx=scales_yx, padding="max_detections", max_detections=frcnn_cfg.max_detections, return_tensors="pt")
        
#                 # Save output_dict to disk
#                 file_without_extension = file.replace(".jpg", "")
#                 save_path = os.path.join(valid_save_path, file)
#                 with open(save_path + ".pickle", "wb") as f:
#                     pickle.dump(output_dict, f)
    print("Done!")
    
if __name__ == "__main__":
    compute_features()