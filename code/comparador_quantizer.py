import numpy as np
import tensorflow as tf
import torch

import quantizer
import quantizer_torch




numpy_array = np.random.rand(16,64,12,12)

tensorflow_tensor = tf.convert_to_tensor(numpy_array, np.float32)
torch_tensor = torch.from_numpy(numpy_array)

tf.InteractiveSession()
out_tensorflow = quantizer.quantize(tensorflow_tensor, tf.convert_to_tensor((-1, 1), np.float32), 1)[0].eval()
out_torch = quantizer_torch.quantize(torch_tensor, torch.from_numpy(np.asarray((-1, 1))), 1).numpy()
comparison =  out_tensorflow == out_torch
print( comparison.all())