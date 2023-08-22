import os
import torch

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

DATA_TYPE_IMAGE = 'img'
DATA_TYPE_TEXT = 'txt'
DATA_TYPES = [DATA_TYPE_IMAGE, DATA_TYPE_TEXT]

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

TEXT_EXTENSIONS = {'txt'}

from .psnr import calc_psnr



parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=32,
                    help='Batch size to use')
parser.add_argument('--clip-model', type=str, default='ViT-B/32',
                    help='CLIP model to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                            'Defaults to `min(8, num_cpus)`'))  
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--data_A_type', type=str, default='img',
                    help=('The modality of data A path. '
                            'Default to img'))
parser.add_argument('--data_B_type', type=str, default='txt',
                    help=('The modality of data B path. '
                            'Default to txt'))
parser.add_argument('--data_A_path', type=str,
                    help=('Paths to the generated images or '
                            'to .npz statistic files'))
parser.add_argument('--data_B_path', type=str,    
                    help=('Paths to the generated images or '
                            'to .npz statistic files'))
parser.add_argument('--fname_strict_mode', action='store_true',
                    help=('If True, the file name of data A and data B must be the same. '
                            'Default to True'))
parser.add_argument('--metric_name', type=str, default='psnr',
                    help=('The metric name to use. '
                            'Default to psnr, allowed: psnr, ssim, lpips, clip'))
parser.add_argument('--save_to', type=str,
                    help=('The path to save the result. '
                            'If not specified, default to data_A_path'))
parser.add_argument('--save_name', type=str, default='image_evaluator_result.txt',
                    help=('The name of the file to save the result. '
                            'Default to image_evaluator_result.txt'))
parser.add_argument('--img_size', type=int, default=256,
                    help=('The image size to use. '
                            'Default to 256'))



def main():
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        args.device = torch.device(args.device)
    
    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()
        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
        args.num_workers = num_workers

    if args.save_to is None:
        args.save_to = args.data_A_path


    if args.metric_name == 'clip':
        # calculate CLIP Score
        pass
    elif args.metric_name == 'psnr':
        # calculate PSNR
        assert args.data_A_type == DATA_TYPE_IMAGE and args.data_B_type == DATA_TYPE_IMAGE, \
            'PSNR only support image modality. However, get {} and {}'.format(
                args.data_A_type, args.data_B_type
            )
        assert args.fname_strict_mode, \
            'PSNR only support fname_strict_mode=True. However, get {}'.format(
                args.fname_strict_mode
            )
        
        # args.save_name = 'psnr.txt'
        # args.img_size = 256
        
        psnr = calc_psnr(args)
        
        with open(os.path.join(args.save_to, args.save_name), 'w') as f:
            f.write('PSNR: {}'.format(psnr))


if __name__ == '__main__':
    # python image_quality_evaluator.py --data_A_path coco/subset_train --data_B_path coco/subset_train_2 --data_A_type img --data_B_type img --fname_strict_mode --metric_name psnr --save_to coco/subset_train
    main()
    