import torch
import torchvision.transforms as T
import numpy as np
from .datasets.image_text_dataset import ImageTextDataset
from .image_quality_evaluator import DATA_TYPE_IMAGE, tqdm


def _calc_psnr(img1, img2):
    """Calculate PSNR between two images.

    Args:
        img1 (np.ndarray): Image 1.
        img2 (np.ndarray): Image 2.

    Returns:
        float: PSNR value.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def _calc_psnr_tensor(img1, img2):
    """Calculate PSNR between two images.

    Args:
        img1 (batch of torch.tensor): Image 1. [between 0. and 1.], [B, C, H, W]
        img2 (batch of torch.tensor): Image 2.

    Returns:
        float: PSNR value.
    """
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3)) # [B]
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calc_psnr(args):
    """Calculate PSNR between two images.

    Args:
        args.data_A_path (str): Path of image 1s.
        args.data_B_path (str): Path of image 2s.
        args.data_A_type (str): 'img'.
        args.data_B_type (str): 'img'.


    Returns:
        float: PSNR value.
    """
    # map PIL image to tensor
    transforms = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor()
    ])

    dataset = ImageTextDataset(
        A_root_dir=args.data_A_path,
        B_root_dir=args.data_B_path,
        A_data_type=DATA_TYPE_IMAGE,
        B_data_type=DATA_TYPE_IMAGE,
        fname_strict_mode=args.fname_strict_mode,
        transform=transforms,
        tokenizer=None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    dataloader = tqdm(dataloader)
    # ----------------- load model -----------------
    psnr = []
    for i, batch in enumerate(dataloader):
        img1 = batch['A'].to(args.device)
        img2 = batch['B'].to(args.device)

        psnr.extend(_calc_psnr_tensor(img1, img2).cpu().numpy())
    
    return np.mean(psnr)