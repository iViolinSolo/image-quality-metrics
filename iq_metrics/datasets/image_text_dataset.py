
import os
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageTextDataset(Dataset):
    '''
    return text and image pair, 
    specifically <image, image>, <text, text>, <image, text> or <text, image>
    
    each sample is a dict:
    {
        'A': image or text tensor
        'B': image or text tensor
        'A_fname': file name of A (without extension)
        'plain_txt': plain text of 'txt' data type (serve as a prompt) # if no text type, then None, if two texts, then a tuple
    }
    '''
    
    TYPES = ['img', 'txt']
    def __init__(self, 
                 A_root_dir, 
                 B_root_dir,
                 A_data_type: str = 'img',
                 B_data_type: str = 'txt',
                 fname_strict_mode: bool = True,
                 transform = None,
                 tokenizer = None) -> None:
        super().__init__()
        assert A_data_type in self.TYPES and B_data_type in self.TYPES, \
            'CLIP Score only support modality of {}. However, get {} and {}'.format(
                self.TYPES, A_data_type, B_data_type
            )
        self.A_data_paths = self._combine_without_prefix(A_root_dir)
        self.A_data_type = A_data_type
        self.B_data_paths = self._combine_without_prefix(B_root_dir)
        self.B_data_type = B_data_type
        self.transform = transform
        self.tokenizer = tokenizer
        if fname_strict_mode:
            # strictly check the file name, e.g. 0001.jpg and 0001.txt
            assert self._check()

    def __len__(self):
        return len(self.A_data_paths)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        path_data_A = self.A_data_paths[index]
        path_data_B = self.B_data_paths[index]
        data_A = self._load_modality(path_data_A, self.A_data_type)
        data_B = self._load_modality(path_data_B, self.B_data_type)

        if self.A_data_type == 'txt' and self.B_data_type == 'txt':
            plain_txt = (data_A[1], data_B[1])
            data_A, data_B = data_A[0], data_B[0]
        elif self.A_data_type == 'txt':
            data_A, plain_txt = data_A
        elif self.B_data_type == 'txt':
            data_B, plain_txt = data_B
        else:
            plain_txt = 'None'

        fname = osp.basename(path_data_A).split('.')[0]
        sample = dict(A=data_A, B=data_B, A_fname=fname, plain_txt=plain_txt)
        return sample
    
    def _load_modality(self, path, modality):
        if modality == 'img':
            data = self._load_img(path)
        elif modality == 'txt':
            data = self._load_txt(path)
        else:
            raise TypeError("Got unexpected modality: {}".format(modality))
        return data

    def _load_img(self, path):
        img = Image.open(path).convert('RGB') # convert to RGB to avoid alpha channel
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def _load_txt(self, path):
        with open(path, 'r') as fp:
            data = fp.read()
            fp.close()
        plain_txt = data
        if self.tokenizer is not None:
            data = self.tokenizer(data).squeeze()
        return data, plain_txt

    def _check(self):
        for idx in range(len(self)):
            # real_name = self.A_data_paths[idx].split('.')
            real_name = osp.basename(self.A_data_paths[idx]).split('.')[0]
            # fake_name = self.B_data_paths[idx].split('.')
            fake_name = osp.basename(self.B_data_paths[idx]).split('.')[0]
            # print(real_name, fake_name)
            if fake_name != real_name:
                return False
        return True
    
    def _combine_without_prefix(self, folder_path, prefix='.'):
        folder = []
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            folder.append(osp.join(folder_path, name))
        folder.sort()
        return folder

