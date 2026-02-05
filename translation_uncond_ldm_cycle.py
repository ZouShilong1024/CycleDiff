import numpy as np
import yaml
import argparse
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from ddm.utils import *
import torchvision as tv
from ddm.encoder_decoder import AutoencoderKL
# from denoising_diffusion_pytorch.transmodel import TransModel
from ddm.data import *
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from fvcore.common.config import CfgNode
from scipy import integrate
from util.eval_score import fid_l2_psnr_ssim

def parse_args():
    parser = argparse.ArgumentParser(description="training vae configure")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, required=True)
    # parser.add_argument("")
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args


def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf

def main(args):
    cfg = CfgNode(args.cfg)
    torch.manual_seed(42)
    np.random.seed(42)

    model_cfg1 = cfg.model1
    assert model_cfg1.ldm, 'This file is only used for ldm！'
    first_stage_cfg1 = model_cfg1.first_stage
    first_stage_model1 = construct_class_by_name(**first_stage_cfg1)
    unet_cfg1 = model_cfg1.unet
    unet1 = construct_class_by_name(**unet_cfg1)
    model_kwargs1 = {'model': unet1, 'auto_encoder': first_stage_model1, 'cfg': model_cfg1}
    model_kwargs1.update(model_cfg1)
    ldm1 = construct_class_by_name(**model_kwargs1)
    model_kwargs1.pop('model')
    model_kwargs1.pop('auto_encoder')

    model_cfg2 = cfg.model2
    first_stage_cfg2 = model_cfg2.first_stage
    first_stage_model2 = construct_class_by_name(**first_stage_cfg2)
    unet_cfg2 = model_cfg2.unet
    unet2 = construct_class_by_name(**unet_cfg2)
    model_kwargs2 = {'model': unet2, 'auto_encoder': first_stage_model2, 'cfg': model_cfg2}
    model_kwargs2.update(model_cfg2)
    ldm2 = construct_class_by_name(**model_kwargs2)
    model_kwargs2.pop('model')
    model_kwargs2.pop('auto_encoder')

    net_G_A_cfg = cfg.net_G
    net_G_A = construct_class_by_name(**net_G_A_cfg)
    net_G_B_cfg = cfg.net_G
    net_G_B = construct_class_by_name(**net_G_B_cfg)

    if cfg.sampler.task == "cat2dog":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "dog2cat":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "wild2dog":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "dog2wild":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "male2female":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "female2male":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "sem2rgb":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "rgb2sem":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "edge2rgb":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "rgb2edge":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "depth2rgb":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "rgb2depth":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "summer2winter":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "winter2summer":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "horse2zebra":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "zebra2horse":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "young2old":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "old2young":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "map2satellite":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "satellite2map":
        data_cfg = cfg.data_test2
    elif cfg.sampler.task == "label2cityscape":
        data_cfg = cfg.data_test
    elif cfg.sampler.task == "cityscape2label":
        data_cfg = cfg.data_test2

    dataset = construct_class_by_name(**data_cfg)
    dl = DataLoader(dataset, batch_size=cfg.sampler.batch_size, shuffle=False, pin_memory=True,
                    num_workers=data_cfg.get('num_workers', 2))

    sampler_cfg = cfg.sampler
    sampler = Sampler(
        ldm1, ldm2, net_G_A, net_G_B, dl, batch_size=sampler_cfg.batch_size,
        results_folder=sampler_cfg.save_folder,cfg=cfg,
    )
    sampler.sample()
    if sampler_cfg.get('cal_metrics', False):
        sampler.cal_metrics(task=sampler_cfg.task, source_gt_path=sampler_cfg.source_gt_path, target_gt_path=sampler_cfg.target_gt_path)
    pass


class Sampler(object):
    def __init__(
            self,
            model1,
            model2,
            net_G_A,
            net_G_B,
            data_loader,
            batch_size=16,
            results_folder='./results',
            rk45=False,
            cfg={},
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision='no',
            kwargs_handlers=[ddp_handler],
        )
        self.model1 = model1
        self.model2 = model2
        self.net_G_A = net_G_A
        self.net_G_B = net_G_B
        self.rk45 = rk45

        self.batch_size = batch_size

        self.image_size = model1.image_size

        dl = self.accelerator.prepare(data_loader)
        self.dl = dl
        self.cfg = cfg
        self.results_folder = Path(results_folder)
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True, parents=True)

        self.model1, self.model2, self.net_G_A, self.net_G_B = self.accelerator.prepare(self.model1, self.model2, self.net_G_A, self.net_G_B)
        data = safe_torch_load(cfg.sampler.ckpt_path,
                          map_location=lambda storage, loc: storage)

        self.model1 = self.accelerator.unwrap_model(self.model1)
        self.model2 = self.accelerator.unwrap_model(self.model2)
        self.net_G_A = self.accelerator.unwrap_model(self.net_G_A)
        self.net_G_B = self.accelerator.unwrap_model(self.net_G_B)

        if cfg.sampler.use_ema:
            sd_d1 = data['ema_d1']
            new_sd = {}
            for k in sd_d1.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]  # remove ema_model.
                    new_sd[new_k] = sd_d1[k]
            sd_d1 = new_sd
            self.model1.load_state_dict(sd_d1)
            sd_d2 = data['ema_d2']
            new_sd = {}
            for k in sd_d2.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]  # remove ema_model.
                    new_sd[new_k] = sd_d2[k]
            sd_d2 = new_sd
            self.model2.load_state_dict(sd_d2)

            sd_G_A = data['ema_G_A']
            new_sd = {}
            for k in sd_G_A.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]  # remove ema_model.
                    new_sd[new_k] = sd_G_A[k]
            sd_G_A = new_sd
            self.net_G_A.load_state_dict(sd_G_A)

            sd_G_B = data['ema_G_B']
            new_sd = {}
            for k in sd_G_B.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]  # remove ema_model.
                    new_sd[new_k] = sd_G_B[k]
            sd_G_B = new_sd
            self.net_G_B.load_state_dict(sd_G_B)
        else:
            self.model1.load_state_dict(data['model1'])
            self.model2.load_state_dict(data['model2'])
            self.net_G_A.load_state_dict(data['net_G_A'])
            self.net_G_B.load_state_dict(data['net_G_B'])
        if 'scale_factor' in data['model1']:
            self.model1.scale_factor = data['model1']['scale_factor']
            self.model2.scale_factor = data['model2']['scale_factor']

    def get_latent_space(self, x, tag=None):
        if "src" in tag:
            z = self.model1.first_stage_model.encode(x)
            z = self.model1.get_first_stage_encoding(z)
            z = self.model1.scale_factor * z
        elif "trg" in tag:
            z = self.model2.first_stage_model.encode(x)
            z = self.model2.get_first_stage_encoding(z)
            z = self.model2.scale_factor * z
        return z

    def sample(self):
        accelerator = self.accelerator
        device = accelerator.device
        with torch.no_grad():
            self.model1.eval()
            self.model2.eval()
            self.net_G_A.eval()
            self.net_G_B.eval()

            for idx, batch in tqdm(enumerate(self.dl), total=len(self.dl)):
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key].to(device)

                if "cat2dog" in self.cfg.sampler.task or "wild2dog" in self.cfg.sampler.task or "male2female" in self.cfg.sampler.task or "sem2rgb" in self.cfg.sampler.task or\
                    "depth2rgb" in self.cfg.sampler.task or "edge2rgb" in self.cfg.sampler.task or "summer2winter" in self.cfg.sampler.task or "horse2zebra" in self.cfg.sampler.task or \
                    "young2old" in self.cfg.sampler.task or "map2satellite" in self.cfg.sampler.task or "label2cityscape" in self.cfg.sampler.task:
                    src_img = batch["image"]
                    x_s = self.get_latent_space(src_img, tag="src_img")

                    c_list, noise = self.model1.reverse_q_sample_c_list_concat(src_img)
                    target_input = []

                    step = 1. / self.model1.sampling_timesteps
                    rho = 1.
                    step_indices = torch.arange(self.model1.sampling_timesteps, dtype=torch.float32, device=device)
                    t_steps = (self.model1.sigma_max ** (1 / rho) + step_indices / (self.model1.sampling_timesteps - 1) * (
                            step - self.model1.sigma_max ** (1 / rho))) ** rho
                    t_steps = reversed(torch.cat([t_steps, torch.zeros_like(t_steps[:1])]))

                    for i in range(len(c_list[:-1])):
                        target_input.append(self.net_G_A(c_list[i], t_steps[i+1].repeat((x_s.shape[0],))))

                    target_input.append(self.net_G_A(c_list[-1], t_steps[-1].repeat((x_s.shape[0],))))
                    target_input.append(noise)
                    pred_img = self.model2.sample_from_c_list(batch_size=src_img.shape[0], c_list=target_input)

                else:
                    trg_img = batch["image"]
                    x_t = self.get_latent_space(trg_img, tag="trg_img")

                    c_list2, noise2 = self.model2.reverse_q_sample_c_list_concat(trg_img)
                    target_input = []

                    step = 1. / self.model2.sampling_timesteps
                    rho = 1.
                    step_indices = torch.arange(self.model2.sampling_timesteps, dtype=torch.float32, device=device)
                    t_steps = (self.model2.sigma_max ** (1 / rho) + step_indices / (self.model2.sampling_timesteps - 1) * (
                            step - self.model2.sigma_max ** (1 / rho))) ** rho
                    t_steps = reversed(torch.cat([t_steps, torch.zeros_like(t_steps[:1])]))

                    for i in range(len(c_list2[:-1])):
                        target_input.append(self.net_G_B(c_list2[i], t_steps[i+1].repeat((x_t.shape[0],))))

                    target_input.append(self.net_G_B(c_list2[-1], t_steps[-1].repeat((x_t.shape[0],))))
                    target_input.append(noise2)
                    pred_img = self.model1.sample_from_c_list(batch_size=trg_img.shape[0], c_list=target_input)

                for j in range(pred_img.shape[0]):
                    img = pred_img[j]
                    file_name = batch["img_name"][j]
                    file_name = self.results_folder / file_name
                    tv.utils.save_image(img, str(file_name)[:-4] + ".png")

        accelerator.print('sampling complete')


    def cal_fid(self, target_path):
        command = 'fidelity -g 0 -f -i -b {} --input1 {} --input2 {}'\
            .format(self.batch_size, str(self.results_folder), target_path)
        os.system(command)

    def cal_metrics(self, task='cat2dog', source_gt_path=None, target_gt_path=None):
        translate_path = self.cfg.sampler.save_folder
        fid_l2_psnr_ssim(task, translate_path, source_gt_path, target_gt_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass
