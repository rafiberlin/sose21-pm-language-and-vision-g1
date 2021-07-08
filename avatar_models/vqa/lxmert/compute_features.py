import pickle
import os 
from tqdm import tqdm
from processing_image import Preprocess
from modeling_frcnn import GeneralizedRCNN
from utils import Config
import torch


def compute_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = "unc-nlp/frcnn-vg-finetuned"
    frcnn_cfg = Config.from_pretrained(checkpoint)
    frcnn_cfg.MODEL.device = device
    print("Loading FRCNN...")
    frcnn = GeneralizedRCNN.from_pretrained(checkpoint, config=frcnn_cfg)
    image_preprocess = Preprocess(frcnn_cfg)
    
    print("Computing features from training images...")
    # Create folder to save featers into
    train_save_path = "precomputed_features/training"
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)

    for root, dirs, files in tqdm(os.walk("/data/ImageCorpora/ADE20K_2016_07_26/images/training")): #edit path
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
    

compute_features()