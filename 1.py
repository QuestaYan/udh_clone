# encoding: utf-8
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.utils as vutils
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import importlib
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="test", help='test dataset')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--imageSize', type=int, default=256, help='the size of images')
parser.add_argument('--testPics', default='./模型文件/', help='folder to output test images')
parser.add_argument('--outlogs', default='./test_logs/', help='folder to output logs')
parser.add_argument('--checkpoint', default='模型文件/autoencoder对比结果/2025-06-04_H01-13-26_256_1_1_120_3_batch_l2_0.75_1colorIn1color_main_udh/checkPoints/best_checkpoint.pth.tar', help='checkpoint address')
parser.add_argument('--norm', default='batch', help='batch or instance')
parser.add_argument('--loss', default='l2', help='l1 or l2')
parser.add_argument('--num_secret', type=int, default=1, help='number of secret images per cover')
parser.add_argument('--num_cover', type=int, default=1, help='number of cover images')
parser.add_argument('--bs_secret', type=int, default=8, help='batch size for secret images')
parser.add_argument('--channel_cover', type=int, default=3, help='1: gray; 3: color')
parser.add_argument('--channel_secret', type=int, default=3, help='1: gray; 3: color')
parser.add_argument('--cover_dependent', type=bool, default=False, help='Whether secret depends on cover')
parser.add_argument('--noise_type', type=str, default='jpeg_compression', help='Type of noise (e.g., gaussian, jpeg, etc.)')
parser.add_argument('--debug', type=bool, default=False, help='debug mode')
parser.add_argument('--num_training', type=int, default=1, help='During training, how many cover images are used for one secret image')

# Noise type mapping for unified parameter handling
noise_class_map = {
    'gaussian_noise': ('GN', {'var': 0.01}),
    'jpeg_compression': ('JpegCompression', {'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')}),
    'identity': ('Identity', {}),
    'gaussian_filter': ('GF', {'sigma': 2}),
    'middle_filter': ('MF', {'kernel': 5}),
    'crop': ('Crop', {'height_ratio': 0.9, 'width_ratio': 0.9}),
    'cropout': ('Cropout', {'height_ratio': 0.5, 'width_ratio': 0.6}),
    'dropout': ('Dropout', {'prob': 0.1}),
    'salt_pepper_noise': ('SP', {'prob': 0.1})
}

# Custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0)

# Print network structure and parameters
def print_network(net, log_path):
    num_params = sum(param.numel() for param in net.parameters())
    print_log(str(net), log_path)
    print_log(f'Total number of parameters: {num_params}', log_path)

# Log printing function
def print_log(log_info, log_path, console=True):
    if console:
        print(log_info)
    if not opt.debug:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a+') as f:
            f.writelines(log_info + '\n')

# Modified: Forward pass to compute rev_secret_img_noised only
def forward_pass(secret_img, cover_img, Hnet, Rnet, criterion, device):
    secret_img, cover_img = secret_img.to(device), cover_img.to(device)
    secret_imgv = secret_img.view(secret_img.size(0) // opt.num_secret, -1, opt.imageSize, opt.imageSize)
    cover_imgv = cover_img.view(cover_img.size(0) // opt.num_cover, -1, opt.imageSize, opt.imageSize)
    H_input = torch.cat((cover_imgv, secret_imgv), dim=1) if opt.cover_dependent else secret_imgv
    container_img = Hnet(H_input)
    if not opt.cover_dependent:
        container_img = container_img + cover_imgv
    errH = criterion(container_img, cover_imgv)
    diffH = (container_img - cover_imgv).abs().mean() * 255

    # Apply noise attack
    if not opt.cover_dependent:
        print(opt.noise_type.upper())
        noise_module = importlib.import_module(f'new_noise.{opt.noise_type}')
        noise_class_name, noise_params = noise_class_map.get(opt.noise_type, (opt.noise_type.upper(), {}))#字典的 get 方法，返回键 opt.noise_type 对应的值。如果键不存在，返回默认值 (opt.noise_type.upper(), {})。
        noise_class = getattr(noise_module, noise_class_name)
        noise_layer = noise_class(**noise_params)
        noised_img = noise_layer([container_img, cover_imgv])
    else:
        noised_img = container_img.clone()

    # Compute rev_secret_img_noised
    rev_secret_img_noised = Rnet(noised_img)
    errR = criterion(rev_secret_img_noised, secret_imgv)
    diffR = (rev_secret_img_noised - secret_imgv).abs().mean() * 255
    return cover_imgv, container_img, noised_img, secret_imgv, rev_secret_img_noised, errH, errR, diffH, diffR

# Modified: Analysis function to use rev_secret_img_noised only
def analysis(test_loader, Hnet, Rnet, criterion, log_path, device):
    print("Starting analysis test...")
    Hnet.eval()
    Rnet.eval()
    import warnings
    warnings.filterwarnings("ignore")

    for i, ((secret_img, _), (cover_img, _)) in enumerate(test_loader, 0):
        # Cover Agnostic Test with noise
        cover_imgv, container_img, noised_img, secret_imgv, rev_secret_img_noised, errH, errR, diffH, diffR = forward_pass(
            secret_img, cover_img, Hnet, Rnet, criterion, device
        )

        # Save qualitative results
        save_result_pic_analysis(
            opt.bs_secret * opt.num_training, cover_imgv, container_img,
            secret_imgv, rev_secret_img_noised, i, opt.testPics
        )

        # Compute metrics
        N = cover_imgv.shape[0]
        
        cover_img_numpy = cover_imgv.clone().cpu().detach().numpy()
        container_img_numpy = container_img.clone().cpu().detach().numpy()
        secret_img_numpy = secret_imgv.cpu().detach().numpy()        
        rev_secret_noised_numpy = rev_secret_img_noised.cpu().detach().numpy()  
        
        cover_img_numpy = cover_img_numpy.transpose(0, 2, 3, 1)
        container_img_numpy = container_img_numpy.transpose(0, 2, 3, 1)
        # noised_img_numpy = noised_img.cpu().numpy().transpose(0, 2, 3, 1)
        secret_img_numpy = secret_img_numpy.transpose(0, 2, 3, 1)
        rev_secret_noised_numpy = rev_secret_noised_numpy.transpose(0, 2, 3, 1)

        print("Cover Agnostic")
        print(f"Secret APD C: {diffH.item():.4f}")
        psnr_c = np.mean([PSNR(cover_img_numpy[i], container_img_numpy[i]) for i in range(N)])
        print(f"Avg. PSNR C: {psnr_c:.4f}")
        ssim_c = np.mean([SSIM(cover_img_numpy[i], container_img_numpy[i], channel_axis=-1, data_range=1.0) for i in range(N)])
        print(f"Avg. SSIM C: {ssim_c:.4f}")
        import PerceptualSimilarity.models
        model = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=device.type == 'cuda')
        lpips_c = model.forward(cover_imgv, container_img).mean().item()
        print(f"Avg. LPIPS C: {lpips_c:.4f}")
        print(f"Secret APD S (Noised): {diffR.item():.4f}")
        psnr_s_noised = np.mean([PSNR(secret_img_numpy[i], rev_secret_noised_numpy[i]) for i in range(N)])
        print(f"Avg. PSNR S (Noised): {psnr_s_noised:.4f}")
        ssim_s_noised = np.mean([SSIM(secret_img_numpy[i], rev_secret_noised_numpy[i], channel_axis=-1, data_range=1.0) for i in range(N)])
        print(f"Avg. SSIM S (Noised): {ssim_s_noised:.4f}")
        lpips_s_noised = model.forward(secret_imgv, rev_secret_img_noised).mean().item()
        print(f"Avg. LPIPS S (Noised): {lpips_s_noised:.4f}")

        # # Cover Agnostic S'
        # cover_img_zero = cover_img.clone().fill_(0.0)
        # cover_imgv_s, container_img_s, noised_img_s, secret_imgv_s, rev_secret_img_noised_s, errH_s, errR_s, diffH_s, diffR_s = forward_pass(
        #     secret_img, cover_img_zero, Hnet, Rnet, criterion, device
        # )

        # secret_img_numpy_s = secret_imgv_s.cpu().numpy().transpose(0, 2, 3, 1)
        # rev_secret_noised_numpy_s = rev_secret_img_noised_s.cpu().numpy().transpose(0, 2, 3, 1)

        # print("\nCover Agnostic S'")
        # print(f"Secret APD S' (Noised): {diffR_s.item():.4f}")
        # psnr_s_prime = np.mean([PSNR(secret_img_numpy_s[i], rev_secret_noised_numpy_s[i]) for i in range(N)])
        # print(f"Avg. PSNR S' (Noised): {psnr_s_prime:.4f}")
        # ssim_s_prime = np.mean([SSIM(secret_img_numpy_s[i], rev_secret_noised_numpy_s[i], channel_axis=-1, data_range=1.0) for i in range(N)])
        # print(f"Avg. SSIM S' (Noised): {ssim_s_prime:.4f}")
        # lpips_s_prime = model.forward(secret_imgv_s, rev_secret_img_noised_s).mean().item()
        # print(f"Avg. LPIPS S' (Noised): {lpips_s_prime:.4f}")

        break  # Process only one batch for testing

# Modified: Updated visualization to include noised_img and rev_secret_img_noised
def save_result_pic_analysis(bs_secret_times_num_training, cover, container, secret, rev_secret,  i, save_path=None):
    path = os.path.join(save_path, 'qualitative_results')
    os.makedirs(path, exist_ok=True)
    resultImgName = os.path.join(path, 'universal_qualitative_results.png')  
    # path = './qualitative_results/'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # resultImgName = path + 'universal_qualitative_results.png'

    cover = cover[:4]
    container = container[:4]
    secret = secret[:4]
    rev_secret = rev_secret[:4]

    cover_gap = container - cover
    secret_gap = rev_secret - secret
    cover_gap = (cover_gap*10 + 0.5).clamp_(0.0, 1.0)
    secret_gap = (secret_gap*10 + 0.5).clamp_(0.0, 1.0)

    showCover = torch.cat((cover, container, cover_gap),0)
    showSecret = torch.cat((secret, rev_secret, secret_gap),0)

    showAll = torch.cat((showCover, showSecret),0)
    size = opt.imageSize
    showAll = showAll.reshape(6, 4, 3, size, size)
    showAll = showAll.permute(1, 0, 2, 3, 4)
    showAll = showAll.reshape(4*6, 3, size, size)
    vutils.save_image(showAll, resultImgName, nrow=6, padding=1, normalize=False)

def main():
    global opt, logPath, DATA_DIR
    opt = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up logging
    if not opt.debug:
        experiment_dir = f"test_{opt.noise_type}_{opt.imageSize}"
        opt.testPics = os.path.join(opt.testPics, experiment_dir)
        opt.outlogs = os.path.join(opt.outlogs, experiment_dir)
        os.makedirs(opt.testPics, exist_ok=True)
        os.makedirs(opt.outlogs, exist_ok=True)
    logPath = os.path.join(opt.outlogs, f'{opt.dataset}_log.txt')
    if opt.debug:
        logPath = './debug/debug_log.txt'
    print_log(str(opt), logPath)

    # Load dataset
    DATA_DIR = '/mnt/d/0important/0important/run_code/a_small_dataset/smallcoco'
    testdir = os.path.join(DATA_DIR, 'val')
    transforms_color = transforms.Compose([
        transforms.Resize([opt.imageSize, opt.imageSize]),
        transforms.ToTensor(),
    ])
    transforms_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([opt.imageSize, opt.imageSize]),
        transforms.ToTensor(),
    ])
    transforms_cover = transforms_gray if opt.channel_cover == 1 else transforms_color
    transforms_secret = transforms_gray if opt.channel_secret == 1 else transforms_color

    test_dataset_cover = ImageFolder(testdir, transforms_cover)
    test_dataset_secret = ImageFolder(testdir, transforms_secret)
    assert test_dataset_cover and test_dataset_secret

    test_loader_secret = DataLoader(test_dataset_secret, batch_size=opt.bs_secret * opt.num_secret,
                                    shuffle=True, num_workers=int(opt.workers))
    test_loader_cover = DataLoader(test_dataset_cover, batch_size=opt.bs_secret * opt.num_cover,
                                   shuffle=True, num_workers=int(opt.workers))
    test_loader = zip(test_loader_secret, test_loader_cover)

    # Initialize networks
    num_downs = 5
    norm_layer = nn.InstanceNorm2d if opt.norm == 'instance' else nn.BatchNorm2d if opt.norm == 'batch' else None
    from models.HidingUNet import UnetGenerator
    from models.RevealNet import RevealNet
    Hnet = UnetGenerator(
        input_nc=opt.channel_secret * opt.num_secret + (opt.channel_cover * opt.num_cover if opt.cover_dependent else 0),
        output_nc=opt.channel_cover * opt.num_cover, num_downs=num_downs, norm_layer=norm_layer,
        output_function=nn.Tanh if not opt.cover_dependent else nn.Sigmoid
    ).cuda()
    Rnet = RevealNet(
        input_nc=opt.channel_cover * opt.num_cover, output_nc=opt.channel_secret * opt.num_secret,
        nhf=64, norm_layer=norm_layer, output_function=nn.Sigmoid
    ).cuda()

    Hnet.apply(weights_init)
    Rnet.apply(weights_init)
    Hnet = torch.nn.DataParallel(Hnet).cuda()
    Rnet = torch.nn.DataParallel(Rnet).cuda()

    # Load checkpoint
    if opt.checkpoint != "":
        checkpoint = torch.load(opt.checkpoint)
        Hnet.load_state_dict(checkpoint['H_state_dict'])
        Rnet.load_state_dict(checkpoint['R_state_dict'])
    else:
        print("WARNING: No checkpoint provided, using initialized weights.")

    print_network(Hnet, logPath)
    print_network(Rnet, logPath)

    # Set loss function
    criterion = nn.MSELoss().to(device) if opt.loss == 'l2' else nn.L1Loss().to(device)

    # Run analysis
    analysis(test_loader, Hnet, Rnet, criterion, logPath, device)

if __name__ == '__main__':
    main()