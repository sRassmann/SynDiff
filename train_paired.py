import argparse
import torch
import numpy as np

import os

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
from dataset import CreateDatasetSynthesis

from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP
import shutil
from skimage.metrics import peak_signal_noise_ratio as psnr
from monai.metrics import SSIMMetric

from backbones.discriminator import Discriminator_small, Discriminator_large
from backbones.ncsnpp_generator_adagn import NCSNpp
import backbones.generator_resnet
from utils.EMA import EMA

from flairsyn.lib.datasets import create_loaders
from flairsyn.lib.utils.visualization import save_grid
from flairsyn.lib.utils.run_utils import setup_ddp
from omegaconf import OmegaConf
from tqdm import tqdm
import torchinfo
from torch.utils.tensorboard import SummaryWriter

torch.multiprocessing.set_sharing_strategy("file_system")


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    # if running in DPP
    if dist.is_initialized():
        for param in params:
            dist.broadcast(param.data, src=0)


# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1.0 - torch.exp(2.0 * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1.0 - eps_small) + eps_small
    return t.to(device)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1.0 - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class Diffusion_Coefficients:
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum**2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = (
        extract(coeff.a_s_cum, t, x_start.shape) * x_start
        + extract(coeff.sigmas_cum, t, x_start.shape) * noise
    )

    return x_t


def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = (
        extract(coeff.a_s, t + 1, x_start.shape) * x_t
        + extract(coeff.sigmas, t + 1, x_start.shape) * noise
    )

    return x_t, x_t_plus_one


# %% posterior sampling
class Posterior_Coefficients:
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (
                torch.tensor([1.0], dtype=torch.float32, device=device),
                self.alphas_cumprod[:-1],
            ),
            0,
        )
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(
            coefficients.posterior_log_variance_clipped, t, x_t.shape
        )
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = 1 - (t == 0).type(torch.float32)

        return (
            mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise
        )

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos


def sample_from_model(coefficients, generator, n_time, x, T, opt, rank=0):
    guidance = x[:, 1:]
    x = x[:, 0:1]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)  # .to(x.device)
            input = torch.cat((x, guidance), axis=1)
            x_0 = generator(input, t_time, latent_z)[:, 0:1]
            x = sample_posterior(coefficients, x_0, x, t)
    return x


config = OmegaConf.load("defaults.yml")["data"]


def train_syndiff(args):
    rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{rank}")

    batch_size = args.batch_size

    target_sequence = config["target_sequence"]
    guidance_sequence = config["guidance_sequences"]
    args.num_channels = len(guidance_sequence) + 1

    nz = args.nz  # latent dimension

    if rank == 0:
        os.makedirs(os.path.join(args.output_path, args.exp, "tb"), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_path, args.exp, "tb"))

    config.batch_size = batch_size
    train_loader, val_loader = create_loaders(**config)

    val_l1_loss = np.zeros([2, args.num_epoch, len(val_loader)])
    val_psnr_values = np.zeros([2, args.num_epoch, len(val_loader)])
    val_ssim_values = np.zeros([2, args.num_epoch, len(val_loader)])
    to_range_0_1 = lambda x: (x + 1.0) / 2.0
    ssim = SSIMMetric(spatial_dims=2)

    # networks performing reverse denoising
    gen_diffusive_1 = NCSNpp(args).to(device)
    # print("Diffusive generator")
    # torchinfo.summary(gen_diffusive_1, input_size=(batch_size, 2, 256, 256))

    disc_diffusive_1 = Discriminator_large(
        nc=2, ngf=args.ngf, t_emb_dim=args.t_emb_dim, act=nn.LeakyReLU(0.2)
    ).to(device)
    # print("Diffusive discriminator")
    # torchinfo.summary(disc_diffusive_1, input_size=(batch_size, 2, 256, 256))

    if world_size > 1:
        gen_diffusive_1 = DDP(gen_diffusive_1, device_ids=[rank])
        disc_diffusive_1 = DDP(disc_diffusive_1, device_ids=[rank])

    optimizer_disc_diffusive_1 = optim.Adam(
        disc_diffusive_1.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2)
    )

    optimizer_gen_diffusive_1 = optim.Adam(
        gen_diffusive_1.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2)
    )

    if args.use_ema:
        optimizer_gen_diffusive_1 = EMA(
            optimizer_gen_diffusive_1, ema_decay=args.ema_decay
        )

    scheduler_gen_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_gen_diffusive_1, args.num_epoch, eta_min=1e-5
    )
    scheduler_disc_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_disc_diffusive_1, args.num_epoch, eta_min=1e-5
    )

    exp = args.exp
    output_path = args.output_path

    exp_path = os.path.join(output_path, exp)
    # version control
    # if rank == 0:
    #     if not os.path.exists(exp_path):
    #         os.makedirs(exp_path)
    #         copy_source(__file__, exp_path)
    #         shutil.copytree("./backbones", os.path.join(exp_path, "backbones"))

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    if args.resume:
        checkpoint_file = os.path.join(exp_path, "content.pth")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint["epoch"]
        epoch = init_epoch
        gen_diffusive_1.load_state_dict(checkpoint["gen_diffusive_1_dict"])
        gen_diffusive_2.load_state_dict(checkpoint["gen_diffusive_2_dict"])

        optimizer_gen_diffusive_1.load_state_dict(
            checkpoint["optimizer_gen_diffusive_1"]
        )
        scheduler_gen_diffusive_1.load_state_dict(
            checkpoint["scheduler_gen_diffusive_1"]
        )
        # load D
        disc_diffusive_1.load_state_dict(checkpoint["disc_diffusive_1_dict"])
        optimizer_disc_diffusive_1.load_state_dict(
            checkpoint["optimizer_disc_diffusive_1"]
        )
        scheduler_disc_diffusive_1.load_state_dict(
            checkpoint["scheduler_disc_diffusive_1"]
        )
        global_step = checkpoint["global_step"]
        print("=> loaded checkpoint (epoch {})".format(checkpoint["epoch"]))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    # torch.autograd.set_detect_anomaly(True)

    val_batch = next(iter(val_loader))
    val_batch = torch.cat([val_batch[seq] for seq in guidance_sequence], dim=1).to(
        device
    )

    for epoch in range(init_epoch, args.num_epoch + 1):
        progress_bar = tqdm(train_loader) if rank == 0 else train_loader
        for iteration, batch in enumerate(progress_bar):
            for p in disc_diffusive_1.parameters():
                p.requires_grad = True

            disc_diffusive_1.zero_grad()

            real_data1 = batch[target_sequence].to(device).float()
            guidance = (
                torch.concat([batch[seq] for seq in guidance_sequence], dim=1)
                .to(device)
                .float()
            )

            # sample t
            t1 = torch.randint(
                0, args.num_timesteps, (real_data1.size(0),), device=device
            )

            # sample x_t and x_tp1
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)
            x1_t.requires_grad = True

            # train discriminator with real
            D1_real = disc_diffusive_1(x1_t, t1, x1_tp1.detach()).view(-1)

            errD1_real = F.softplus(-D1_real)
            errD1_real = errD1_real.mean()

            errD_real = errD1_real
            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None:
                grad1_real = torch.autograd.grad(
                    outputs=D1_real.sum(), inputs=x1_t, create_graph=True
                )[0]
                grad1_penalty = (
                    grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()
                grad_penalty = args.r1_gamma / 2 * grad1_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad1_real = torch.autograd.grad(
                        outputs=D1_real.sum(), inputs=x1_t, create_graph=True
                    )[0]
                    grad1_penalty = (
                        grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2
                    ).mean()
                    grad_penalty = args.r1_gamma / 2 * grad1_penalty
                    grad_penalty.backward()

            # train with fake
            latent_z1 = torch.randn(batch_size, nz, device=device)

            # x_tp1 is concatenated with source contrast and x_0_predict is predicted
            x1_0_predict_diff = gen_diffusive_1(
                torch.cat((x1_tp1.detach(), guidance), axis=1),  # useless detach
                t1,
                latent_z1,
            )
            # sampling q(x_t | x_0_predict, x_t+1)
            x1_pos_sample = sample_posterior(
                pos_coeff, x1_0_predict_diff[:, [0], :], x1_tp1, t1
            )

            # D output for fake sample x_pos_sample
            output1 = disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)

            errD1_fake = F.softplus(output1)
            errD_fake = errD1_fake.mean()
            errD_fake.backward()

            errD = errD_real + errD_fake

            # Update D
            optimizer_disc_diffusive_1.step()

            # G part
            for p in disc_diffusive_1.parameters():
                p.requires_grad = False

            gen_diffusive_1.zero_grad()

            t1 = torch.randint(
                0, args.num_timesteps, (real_data1.size(0),), device=device
            )
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)

            x1_0_predict_diff = gen_diffusive_1(
                torch.cat((x1_tp1, guidance), axis=1), t1, latent_z1
            )
            # sampling q(x_t | x_0_predict, x_t+1)
            x1_pos_sample = sample_posterior(
                pos_coeff, x1_0_predict_diff[:, [0], :], x1_tp1, t1
            )
            output1 = disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)

            errG1 = F.softplus(-output1)
            errG1 = errG1.mean()
            errG_adv = errG1

            # L1 loss
            errG1_L1 = F.l1_loss(x1_0_predict_diff[:, [0], :], real_data1)
            errG_L1 = errG1_L1

            errG = errG_adv + args.lambda_l1_loss * errG_L1
            errG.backward()

            optimizer_gen_diffusive_1.step()

            global_step += 1
            if rank == 0:
                # use writer
                writer.add_scalar("G-Adv", errG_adv.item(), global_step)
                writer.add_scalar("G-L1", errG_L1.item(), global_step)
                writer.add_scalar("D Loss", errD.item(), global_step)

        if not args.no_lr_decay:
            scheduler_gen_diffusive_1.step()
            scheduler_disc_diffusive_1.step()

            # if rank == 0:
            #     if epoch % 200 == 0:
            #         torchvision.utils.save_image(
            #             x1_pos_sample,
            #             os.path.join(exp_path, "xpos1_epoch_{}.png".format(epoch)),
            #             normalize=True,
            #         )

        if epoch % args.save_ckpt_every == 0:
            # save images
            # concatenate noise and source contrast
            x1_t = torch.cat((torch.randn_like(real_data1), val_batch), axis=1)
            fake_sample1 = sample_from_model(
                pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t, T, args
            )

            # save model ckpt
            save_grid(
                fake_sample1,
                os.path.join(exp_path, "fake_sample1_{}.png".format(epoch)),
                columns=4,
            )

            if args.use_ema:
                optimizer_gen_diffusive_1.swap_parameters_with_ema(
                    store_params_in_ema=True
                )
            torch.save(
                gen_diffusive_1.state_dict(),
                os.path.join(exp_path, "gen_diffusive_1_{}.pth".format(epoch)),
            )
            if args.use_ema:  # ?
                optimizer_gen_diffusive_1.swap_parameters_with_ema(
                    store_params_in_ema=True
                )

            for iteration, batch in enumerate(val_loader):
                guidance = torch.cat(
                    [batch[seq] for seq in guidance_sequence], dim=1
                ).to(device)
                real_data = batch[target_sequence].to(device).float()

                x1_t = torch.cat((torch.randn_like(real_data), guidance), axis=1)
                fake_sample1 = sample_from_model(
                    pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t, T, args
                )
                fake_sample1 = to_range_0_1(fake_sample1)
                fake_sample1 = fake_sample1 / fake_sample1.mean()
                real_data = to_range_0_1(real_data)
                real_data = real_data / real_data.mean()

                val_ssim_values[0, epoch, iteration] = (
                    ssim(real_data, fake_sample1).mean().item()
                )
                fake_sample1 = fake_sample1.cpu().numpy()
                real_data = real_data.cpu().numpy()
                val_l1_loss[0, epoch, iteration] = abs(fake_sample1 - real_data).mean()

                val_psnr_values[0, epoch, iteration] = psnr(
                    real_data, fake_sample1, data_range=real_data.max()
                )

            if rank == 0:  # a bit stupid to use only rank 0, but it's fine
                writer.add_scalar(
                    f"Val PSNR",
                    np.nanmean(val_psnr_values[0, epoch, :]),
                    epoch,
                )
                writer.add_scalar(
                    f"Val L1 Loss",
                    np.nanmean(val_l1_loss[0, epoch, :]),
                    epoch,
                )
                writer.add_scalar(
                    f"Val SSIM",
                    np.nanmean(val_ssim_values[0, epoch, :]),
                    epoch,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("syndiff parameters")
    parser.add_argument(
        "--seed", type=int, default=1024, help="seed used for initialization"
    )

    parser.add_argument("--resume", action="store_true", default=False)

    parser.add_argument("--image_size", type=int, default=256, help="size of image")
    parser.add_argument(
        "--centered", action="store_false", default=True, help="-1,1 scale"
    )
    parser.add_argument("--use_geometric", action="store_true", default=False)
    parser.add_argument(
        "--beta_min", type=float, default=0.1, help="beta_min for diffusion"
    )
    parser.add_argument(
        "--beta_max", type=float, default=20.0, help="beta_max for diffusion"
    )

    parser.add_argument(
        "--num_channels_dae",
        type=int,
        default=128,
        help="number of initial channels in denosing model",
    )
    parser.add_argument(
        "--n_mlp", type=int, default=3, help="number of mlp layers for z"
    )
    parser.add_argument("--ch_mult", nargs="+", type=int, help="channel multiplier")
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=2,
        help="number of resnet blocks per scale",
    )
    parser.add_argument(
        "--attn_resolutions", default=(16,), help="resolution of applying attention"
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument(
        "--resamp_with_conv",
        action="store_false",
        default=True,
        help="always up/down sampling with conv",
    )
    parser.add_argument(
        "--conditional", action="store_false", default=True, help="noise conditional"
    )
    parser.add_argument("--fir", action="store_false", default=True, help="FIR")
    parser.add_argument("--fir_kernel", default=[1, 3, 3, 1], help="FIR kernel")
    parser.add_argument(
        "--skip_rescale", action="store_false", default=True, help="skip rescale"
    )
    parser.add_argument(
        "--resblock_type",
        default="biggan",
        help="tyle of resnet block, choice in biggan and ddpm",
    )
    parser.add_argument(
        "--progressive",
        type=str,
        default="none",
        choices=["none", "output_skip", "residual"],
        help="progressive type for output",
    )
    parser.add_argument(
        "--progressive_input",
        type=str,
        default="residual",
        choices=["none", "input_skip", "residual"],
        help="progressive type for input",
    )
    parser.add_argument(
        "--progressive_combine",
        type=str,
        default="sum",
        choices=["sum", "cat"],
        help="progressive combine method.",
    )

    parser.add_argument(
        "--embedding_type",
        type=str,
        default="positional",
        choices=["positional", "fourier"],
        help="type of time embedding",
    )
    parser.add_argument(
        "--fourier_scale", type=float, default=16.0, help="scale of fourier transform"
    )
    parser.add_argument("--not_use_tanh", action="store_true", default=False)

    # geenrator and training
    parser.add_argument("--exp", default="ixi_synth", help="name of experiment")
    parser.add_argument("--input_path", help="path to input data")
    parser.add_argument("--output_path", default="output", help="path to output saves")
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--num_timesteps", type=int, default=4)

    parser.add_argument("--z_emb_dim", type=int, default=256)
    parser.add_argument("--t_emb_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1, help="input batch size")
    parser.add_argument("--num_epoch", type=int, default=1200)
    parser.add_argument("--ngf", type=int, default=64)

    parser.add_argument("--lr_g", type=float, default=1.5e-4, help="learning rate g")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="learning rate d")
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.9, help="beta2 for adam")
    parser.add_argument("--no_lr_decay", action="store_true", default=False)

    parser.add_argument(
        "--use_ema", action="store_true", default=False, help="use EMA or not"
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.9999, help="decay rate for EMA"
    )

    parser.add_argument("--r1_gamma", type=float, default=0.05, help="coef for r1 reg")
    parser.add_argument(
        "--lazy_reg", type=int, default=None, help="lazy regulariation."
    )

    parser.add_argument("--save_content", action="store_true", default=False)
    parser.add_argument(
        "--save_content_every",
        type=int,
        default=10,
        help="save content for resuming every x epochs",
    )
    parser.add_argument(
        "--save_ckpt_every", type=int, default=10, help="save ckpt every x epochs"
    )
    parser.add_argument(
        "--lambda_l1_loss",
        type=float,
        default=0.5,
        help="weightening of l1 loss part of diffusion ans cycle models",
    )

    args = parser.parse_args()

    train_syndiff(args)

# run with smth like:
# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 torchrun --nproc_per_node=8 train_paired.py \
# --exp exp_syndiff_bs_96 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 \
# --num_res_blocks 2 --num_epoch 20000 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 \
# --save_ckpt_every 200 --r1_gamma 1. --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 \
# --save_content --output_path output --batch_size 12
