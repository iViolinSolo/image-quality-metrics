


if __name__ == '__main__':
    from iq_metrics.datasets.image_text_dataset import ImageTextDataset
    from iq_metrics.psnr import _calc_psnr_tensor
    import torch
    import torchvision.transforms as T
    # b_pth = '/home/u2280887/GitHub/xmh/coco-val2014/coco/subset'
    a_pth = '/home/u2280887/GitHub/xmh/experiments/outputs/exp1/samples'
    b_pth = '/home/u2280887/GitHub/xmh/experiments/outputs/exp2/samples'
    transforms = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    dataset = ImageTextDataset(a_pth, b_pth, 'img', 'img', True, transforms, None)


    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=8,
        pin_memory=True
    )

    it = iter(dataloader)
    batch  = next(it)

    a0 = batch['A']
    b0 = batch['B']
    psnrs = _calc_psnr_tensor(a0, b0)
    print(psnrs, torch.mean(psnrs))
