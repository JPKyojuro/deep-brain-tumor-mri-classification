from torchvision.utils import save_image
import torch
from common import load_config
from train import MRIDataset

ds = MRIDataset(load_config(), "train", train=True, d8=True)  # no norm, so pixels stay 0-1
idx = [100, 1000, 1800, 2800, 3300, 4000]  # items are sorted by class; roughly 2 glioma/meningioma, 1 no_tumor, 2 pituitary
rows = [torch.stack([ds[i][0] for _ in range(4)]) for i in idx]  # 4 random augmentations per image
save_image(torch.cat(rows), "outputs/setup/aug_d8.png", nrow=4)