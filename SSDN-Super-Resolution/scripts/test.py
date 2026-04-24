import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ssdn import SSDN
from utils.metrics import calculate_psnr, calculate_ssim


def get_args_parser():
    parser = argparse.ArgumentParser(description="Test Script for SSDN")
    parser.add_argument('--model_path', type=str, required=True, help="Path to weights")
    parser.add_argument('--lr_dir', type=str, required=True, help="Directory of LR images")
    parser.add_argument('--hr_dir', type=str, default=None, help="Directory of HR images")
    parser.add_argument('--save_dir', type=str, default='results/', help="Save directory")
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--in_chans', type=int, default=3)
    return parser


def main():
    args = get_args_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    model = SSDN(in_nc=args.in_chans, out_nc=args.in_chans, nf=64, nb=16, upscale=args.scale)
    state_dict = torch.load(args.model_path, map_location=device)
    clean_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict)
    model.to(device)
    model.eval()

    files = sorted([f for f in os.listdir(args.lr_dir) if f.endswith(('.png', '.jpg', '.bmp', '.tif'))])
    psnr_list, ssim_list = [], []

    print(f"Evaluating {len(files)} images...")

    for filename in files:
        img_mode = 'RGB' if args.in_chans == 3 else 'L'
        lr_img = Image.open(os.path.join(args.lr_dir, filename)).convert(img_mode)
        lr_tensor = TF.to_tensor(lr_img).unsqueeze(0).to(device)

        with torch.no_grad():
            sr_tensor = model(lr_tensor).squeeze(0).cpu().clamp(0, 1)

        sr_img = TF.to_pil_image(sr_tensor)
        sr_img.save(os.path.join(args.save_dir, filename))

        if args.hr_dir:
            base_name = os.path.splitext(filename)[0].replace(f'x{args.scale}', '')
            hr_path = os.path.join(args.hr_dir, base_name + os.path.splitext(filename)[1])

            if os.path.exists(hr_path):
                hr_img = Image.open(hr_path).convert(img_mode)
                sr_np, hr_np = np.array(sr_img), np.array(hr_img)

                is_rgb = (args.in_chans == 3)
                cur_psnr = calculate_psnr(sr_np, hr_np, crop_border=args.scale, test_y_channel=is_rgb)
                cur_ssim = calculate_ssim(sr_np, hr_np, crop_border=args.scale, test_y_channel=is_rgb)

                psnr_list.append(cur_psnr)
                ssim_list.append(cur_ssim)
                print(f"[{filename}] PSNR: {cur_psnr:.2f} dB, SSIM: {cur_ssim:.4f}")

    if psnr_list:
        print(f"\n--- Final Results ({len(psnr_list)} images) ---")
        print(f"Average PSNR: {np.mean(psnr_list):.4f} dB")
        print(f"Average SSIM: {np.mean(ssim_list):.4f}")


if __name__ == '__main__':
    main()