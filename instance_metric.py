import argparse
import numpy as np

from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, set_random_seed
from data.cls_to_names import *
from data.imagenet_variants import imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import os

from instance_method.tpt_metric import TPT
from utils.metric import compute_skip_metrics


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def main():
    args = parser.parse_args()
    assert args.gpu is not None

    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    dset = args.test_sets

    args.output_dir = os.path.join(
        args.output_dir,
        args.metric,
        args.arch.replace('/', ''),
        'seed_' + str(args.seed),
        args.algorithm,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    args.out_file = open(
        os.path.join(args.output_dir, 'log_' + dset + '.txt'),
        'w',
        encoding='utf-8',
    )
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    if True:
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                        augmix=len(dset)>1)
        batchsize = 1
        val_dataset = build_dataset(dset, data_transform, args.data, mode=args.dataset_mode)

        print_log = "number of test samples: {}".format(len(val_dataset))
        args.out_file.write(print_log + '\n')
        args.out_file.flush()
        print(print_log+'\n')

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

        print("evaluating: {}".format(dset))

        if len(dset) > 1: 
            classnames = eval("{}_classes".format(dset.lower()))
        else:
            assert dset in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if dset in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(dset.lower()))
                if dset == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all


# 
        if True:
            model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)

            if args.load is not None:
                print("Use pre-trained soft prompt (CoOp) as initialization")
                load_path = {
                    'RN50': 'rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed2/prompt_learner/model.pth.tar-50',
                    'ViT-B/16': 'vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed2/prompt_learner/model.pth.tar-50'
                }[args.arch]
                load_path = os.path.join(args.load, load_path)
                pretrained_ctx = torch.load(load_path, weights_only=False, map_location='cpu')['state_dict']['ctx']
                assert pretrained_ctx.size()[0] == args.n_ctx
                args.pretrained_ctx = pretrained_ctx
            
            # if args.load_tecoa:
            #     print('loading tecoa')
            #     args.robust_pretrain_path = {
            #         'RN50': 'your_cache_path/tecoa/rn50_eps1.pth.tar',
            #         'ViT-B/32': 'your_cache_path/tecoa/vitb32_eps4.pth.tar'
            #     }[args.arch]
            #     robust_state_dict = torch.load(args.robust_pretrain_path, map_location='cpu')
            #     model.image_encoder.load_state_dict(robust_state_dict['vision_encoder_state_dict'])

        print("=> Model created: visual backbone {}".format(args.arch))
        
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
        else:
            assert args.gpu is not None
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        
        if args.algorithm == 'tpt':
            tta_trainer = TPT(model, args)
        else:
            raise ValueError(f"Invalid algorithm: {args.algorithm}")
        
        metrics_list = []
        
        num_pred = 2
        correct_num = np.zeros(num_pred)

        tta_trainer.model.eval()
        pbar = tqdm(val_loader, desc=f'Evaluating {dset}')
        for i, data in enumerate(pbar):

            assert args.gpu is not None
            target = data[1]
            
            if i > 0 and (target != old_target):
                print_log = 'iter:{}/{}'.format(i, len(val_loader))
                args.out_file.write(print_log + '\n')
                args.out_file.flush()

                correct_num = np.zeros(num_pred)

            old_target = target

            images = data[0]
            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = images[k].cuda(args.gpu, non_blocking=True)
                image = images[0]
            else:
                if len(images.size()) > 4:
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                images = images.cuda(args.gpu, non_blocking=True)
                image = images
        
            images = torch.cat(images, dim=0)

            return_dict = tta_trainer.adaptation_process(image, images, args)
            
            metrics_list.append({
                'zs_pred': return_dict['zs_pred'],
                'tta_pred': return_dict['tta_pred'],
                'metric': return_dict['metric'],
                'target': target.item()
            })
            
            acc_zs = int(return_dict['zs_pred'] == target.cpu().item())
            acc_tta = int(return_dict['tta_pred'] == target.cpu().item())
            correct_num[0] += acc_zs
            correct_num[1] += acc_tta

        print_log = 'iter:{}/{}'.format(i, len(val_loader))
        args.out_file.write(print_log + '\n')
        args.out_file.flush()

        del val_dataset, val_loader
    
        
        if len(metrics_list) > 0:
            skip_results = compute_skip_metrics(metrics_list)
            print_log = "=> ZS Acc: {:.2f}%, TTA Acc: {:.2f}%".format(
                skip_results['zs_acc'], skip_results['tta_acc'])
            args.out_file.write(print_log + '\n')
            args.out_file.flush()
            print(print_log+'\n')

            print_log = "=> Acc at different skip rates"
            args.out_file.write(print_log + '\n')
            print(print_log)
            for skip_rate, skip_acc in zip(
                skip_results['skip_rates'], skip_results['skip_accs']
            ):
                print_log = "skip_rate={:.2f}, acc={:.2f}%".format(
                    skip_rate, skip_acc
                )
                args.out_file.write(print_log + '\n')
                print(print_log)
            args.out_file.flush()
            
            print_log = "=> AUC: {:.4f}, AEP: {:.4f}".format(
                skip_results['auc'], skip_results['aep'])
            args.out_file.write(print_log + '\n')
            args.out_file.flush()
            print(print_log+'\n')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect TPT Metrics')
    parser.add_argument('--data', type=str, default='', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop') # ./coop_weight/to_gdrive/
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='temp')

    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')

    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--algorithm', type=str, default='tpt', choices=['tpt'])
    parser.add_argument(
        '--metric',
        type=str,
        default='cas',
        choices=['cas', 'energy', 'mcm', 'random'],
        help='metric used to rank samples for skipping TPT',
    )

    main()

