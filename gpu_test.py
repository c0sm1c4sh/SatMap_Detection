import torch
print('GPU Name:', torch.cuda.get_device_name(0))
print('CUDA Available:', torch.cuda.is_available())