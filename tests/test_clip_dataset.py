import os
import clip
import numpy as np
import torch
from tqdm import tqdm
from iq_metircs.datasets.image_text_dataset import ImageTextDataset


def load_clip_and_calculate_clipscore():
    # ----------------- parameters ----------------- 
    clip_model = 'ViT-B/32'
    batch_size = 32
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # os.sched_getaffinity is not available under Windows, use
        # os.cpu_count instead (which may not return the *available* number
        # of CPUs).
        num_cpus = os.cpu_count()
    num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    # ----------------- parameters ----------------- 

    # ----------------- load model -----------------
    print('Loading CLIP model: {}'.format(clip_model))
    model, preprocess = clip.load(clip_model, device=device)
    print(preprocess)

    dataset = ImageTextDataset(
        A_root_dir='coco/subset_train',
        B_root_dir='coco/subset_train-txts',
        A_data_type='img',
        B_data_type='txt',
        fname_strict_mode=True,
        transform=preprocess,
        tokenizer=clip.tokenize
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    dataloader = tqdm(dataloader)
    # ----------------- load model -----------------

    # ----------------- calculate CLIP Score -----------------

    img_list = []
    txt_list = []
    fname_list = []
    plain_txt_list = []
    clip_score_list = []

    it = iter(dataloader)
    for idx, sample in tqdm(enumerate(it), total=len(dataloader)):
        img = sample['A'].to(device)
        txt = sample['B'].to(device)
        fname = sample['A_fname']  #list
        plain_txt = sample['plain_txt'] #list

        # print(fname, plain_txt)
        with torch.no_grad():
            txt_features = model.encode_text(txt)
            img_features = model.encode_image(img)
            # print(txt_features.shape, img_features.shape) # (50, 768). (50, 768)
            # normalize features
            txt_features = txt_features / txt_features.norm(dim=1, keepdim=True).to(torch.float32)
            img_features = img_features / img_features.norm(dim=1, keepdim=True).to(torch.float32)
            clip_score = (txt_features * img_features).sum(dim=1).cpu().numpy()
        
        img_list.append(img_features.cpu().numpy())
        txt_list.append(txt_features.cpu().numpy())
        fname_list.extend(fname)
        plain_txt_list.extend(plain_txt)
        clip_score_list.extend(clip_score)
        # print('Processing {}/{}'.format(idx, len(dl)))
    
    # img_list = np.concatenate(img_list, axis=0)
    # txt_list = np.concatenate(txt_list, axis=0)
    # print(img_list.shape, txt_list.shape) # (5000, 768). (5000, 768)
    print('Done!')
    print('Calculating CLIP Score:')
    print('CLIP Score: {}'.format(np.mean(clip_score_list)))
    # ----------------- calculate CLIP Score -----------------