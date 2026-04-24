import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ssdn import SSDN
from data.dataset import SRDataset


def get_args_parser():
    parser = argparse.ArgumentParser(description='SSDN Distributed Training Script')

    parser.add_argument('--lr_train_path', type=str, required=True, help='Path to LR training dataset')
    parser.add_argument('--hr_train_path', type=str, required=True, help='Path to HR training dataset')

    parser.add_argument('--epochs', type=int, default=500, help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate')
    parser.add_argument('--patch_size', type=int, default=48, help='LR patch size for cropping')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers per GPU')

    parser.add_argument('--scale', type=int, default=4, help='Super-resolution scale factor')
    parser.add_argument('--in_chans', type=int, default=3, help='Input channels')
    parser.add_argument('--nf', type=int, default=64, help='Number of feature channels')
    parser.add_argument('--nb', type=int, default=16, help='Number of residual blocks')

    parser.add_argument('--checkpoint_dir', type=str, default='experiments/ssdn_weights', help='Output directory')
    return parser


def setup_ddp():
    """Initialize distributed data parallel environment."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Destroy distributed process group."""
    dist.destroy_process_group()


def main():
    args = get_args_parser().parse_args()

    # Initialize Dual-GPU DDP environment
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    is_main_process = (local_rank == 0)

    if is_main_process:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Initializing DDP Training with {world_size} GPUs.")
        print(f"Output directory: {os.path.abspath(args.checkpoint_dir)}")

    # Initialize Model
    model = SSDN(
        in_nc=args.in_chans,
        out_nc=args.in_chans,
        nf=args.nf,
        nb=args.nb,
        upscale=args.scale
    ).to(local_rank)

    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Optimization Setup
    criterion = torch.nn.L1Loss().to(local_rank)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = GradScaler('cuda', enabled=True)

    # Dataset & Distributed Sampler
    train_dataset = SRDataset(
        lr_path=args.lr_train_path,
        hr_path=args.hr_train_path,
        patch_size=args.patch_size,
        scale=args.scale,
        is_train=True
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()

        # Only show progress bar on the main process
        if is_main_process:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        else:
            pbar = train_loader

        for lr, hr in pbar:
            lr = lr.to(local_rank, non_blocking=True)
            hr = hr.to(local_rank, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type='cuda', enabled=False):
                sr_out = model(lr)
                loss = criterion(sr_out, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if is_main_process:
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

        scheduler.step()

        # Checkpointing (Main Process Only)
        if is_main_process and (epoch % 5 == 0 or epoch == args.epochs):
            save_path = os.path.join(args.checkpoint_dir, f'ssdn_epoch_{epoch}.pth')
            torch.save(model.module.state_dict(), save_path)

    cleanup_ddp()


if __name__ == '__main__':
    main()