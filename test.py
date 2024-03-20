import sys

import argparse
import torch
import numpy as np

import os

from backbones.ncsnpp_generator_adagn import NCSNpp

from flairsyn.lib.datasets import get_datasets
from flairsyn.lib.inference import save_output_volume
from omegaconf import OmegaConf
from tqdm import tqdm

from train_paired import sample_from_model

config = OmegaConf.load("defaults.yml")


def psnr(img1, img2):
    # Peak Signal to Noise Ratio

    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(img1.max() / torch.sqrt(mse))


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


def load_checkpoint(checkpoint_dir, netG, name_of_network, epoch, device="cuda:0"):
    checkpoint_file = checkpoint_dir.format(name_of_network, epoch)

    checkpoint = torch.load(checkpoint_file, map_location=device)
    ckpt = checkpoint

    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()

    # from flopth import flopth
    #
    # image = torch.rand(1, 3, 256, 256)
    # timestep = torch.Tensor((0,)).long()
    # latent_z = torch.randn(image.size(0), 100, device=image.device)
    # flops, params = flopth(netG, inputs=(image, timestep, latent_z), show_detail=True)
    # print(f"FLOPs: {flops}, Params: {params}")
    #
    # sys.exit(0)


# %%
def sample_and_test(args):
    target_seq = config.data.target_sequence
    guidance_seqs = config.data.guidance_sequences

    args.num_channels = len(guidance_seqs) + 1

    torch.manual_seed(42)
    device = "cuda:0"

    epoch_chosen = args.which_epoch

    # Initializing and loading network
    gen_diffusive_1 = NCSNpp(args).to(device)

    exp = args.exp
    output_dir = args.output_path
    exp_path = os.path.join(output_dir, exp)

    checkpoint_file = exp_path + "/{}_{}.pth"
    load_checkpoint(
        checkpoint_file,
        gen_diffusive_1,
        "gen_diffusive_1",
        epoch=str(epoch_chosen),
        device=device,
    )

    T = get_time_schedule(args, device)

    pos_coeff = Posterior_Coefficients(args, device)

    _, val = get_datasets(
        dataset=args.dataset_json,
        data_dir=args.data_dir,
        relevant_sequences=guidance_seqs + [target_seq],
        size=None,
        cache=None,
        subset_train=0,
        normalize_to=(-1, 1),
        skull_strip=1,
    )

    # test
    for i, vol in enumerate(tqdm(val)):
        pred_vol = torch.zeros_like(vol[config.data.target_sequence])
        guidance = torch.cat([vol[seq] for seq in guidance_seqs], dim=0).float()
        guidance = guidance.permute(1, 0, 2, 3)

        for j in range(0, vol[target_seq].shape[1], args.batch_size):
            g = guidance[j : j + args.batch_size]
            b, c, h, w = g.shape

            # pad to batch size, 2, 256, 256
            pad = -torch.ones([b, c, 256, 256])
            offset_h = (256 - h) // 2
            offset_w = (256 - w) // 2
            pad[:, :, offset_h : offset_h + h, offset_w : offset_w + w] = g
            pad = pad.to(device)

            noise = torch.randn((b, 1, 256, 256), device=device)
            x1_t = torch.cat([noise, pad], axis=1)
            pred_batch = sample_from_model(
                pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t, T, args
            )
            pred_vol[0, j : j + args.batch_size] = pred_batch[
                :, 0, offset_h : offset_h + h, offset_w : offset_w + w
            ].cpu()

        vol["pred"] = torch.clamp(pred_vol, -1, 1)
        save_output_volume(
            vol,
            output_path=os.path.join(exp_path, args.out_dir_name),
            save_keys=list(config.data.guidance_sequences)
            + ["pred", config.data.target_sequence, "mask"],
            target_sequence=config.data.target_sequence,
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
    parser.add_argument(
        "--which_epoch", type=int, default=20000, help="which epoch to load"
    )

    parser.add_argument(
        "--dataset_json",
        type=str,
        default="../data/RS/RS_train_split.json",
        help="path to the dataset json file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/RS/conformed_mask_reg",
        help="path to the dataset directory",
    )
    parser.add_argument(
        "--out_dir_name",
        type=str,
        default="inference",
        help="name of the output directory",
    )

    args = parser.parse_args()

    sample_and_test(args)
