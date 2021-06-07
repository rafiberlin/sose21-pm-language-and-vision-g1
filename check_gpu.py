import torch
import tensorflow as tf

print("Pytorch GPU Support:", torch.cuda.is_available())

physical_devices = tf.config.list_physical_devices('GPU')
print("Tensorflow", "Num GPUs:", len(physical_devices))
