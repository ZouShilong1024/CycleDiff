import yaml
import argparse
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from ddm.ema import EMA
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
from ddm.loss import *
from ddm.ddm_const import SpecifyGradient2

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

    model_cfg1 = cfg.model1
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

    net_D_A_cfg = cfg.net_D
    net_D_A = construct_class_by_name(**net_D_A_cfg)
    net_D_B_cfg = cfg.net_D
    net_D_B = construct_class_by_name(**net_D_B_cfg)

    data_cfg = cfg.data
    dataset = construct_class_by_name(**data_cfg)
    dl = DataLoader(dataset, batch_size=data_cfg.batch_size, shuffle=True, pin_memory=True,
                    num_workers=data_cfg.get('num_workers', 2))
    train_cfg = cfg.trainer
    trainer = Trainer(
        ldm1, ldm2, net_G_A, net_G_B, net_D_A, net_D_B, dl, train_batch_size=data_cfg.batch_size,
        gradient_accumulate_every=train_cfg.gradient_accumulate_every,
        train_lr=train_cfg.lr, trans_net_lr=train_cfg.trans_net_lr, train_num_steps=train_cfg.train_num_steps,
        save_every=train_cfg.save_every, sample_every=train_cfg.sample_every, results_folder=train_cfg.results_folder,
        amp=train_cfg.amp, fp16=train_cfg.fp16, log_freq=train_cfg.log_freq, cfg=cfg,
        resume_milestone=train_cfg.resume_milestone,
        train_wd=train_cfg.get('weight_decay', 1e-2),
    )
    if train_cfg.test_before:
        if trainer.accelerator.is_main_process:
            with torch.no_grad():
                for datatmp in dl:
                    break
                device = trainer.accelerator.device
                src_img = datatmp["src_img"].to(trainer.accelerator.device)

                x_s = trainer.get_latent_space(src_img, tag="src_img")
                C_S = -1 * x_s

                model1_unwrapped = trainer._model1_unwrapped
                model2_unwrapped = trainer._model2_unwrapped

                c_list, noise = model1_unwrapped.reverse_q_sample_c_list_concat(src_img.to(device))
                target_input = []

                step = 1. / model1_unwrapped.sampling_timesteps
                rho = 1.
                step_indices = torch.arange(model1_unwrapped.sampling_timesteps, dtype=torch.float32, device=device)
                t_steps = (model1_unwrapped.sigma_max ** (1 / rho) + step_indices / (model1_unwrapped.sampling_timesteps - 1) * (
                        step - model1_unwrapped.sigma_max ** (1 / rho))) ** rho
                t_steps = reversed(torch.cat([t_steps, torch.zeros_like(t_steps[:1])]))

                for i in range(len(c_list[:-1])):
                    target_input.append(trainer.net_G_A(c_list[i], t_steps[i+1].repeat((x_s.shape[0],))))

                target_input.append(trainer.net_G_A(c_list[-1], t_steps[-1].repeat((x_s.shape[0],))))
                target_input.append(noise)
                pred_img = model2_unwrapped.sample_from_c_list(batch_size=src_img.shape[0], c_list=target_input)

                pred_model1 = model1_unwrapped.sample(batch_size=src_img.shape[0], )
                pred_model2 = model2_unwrapped.sample(batch_size=src_img.shape[0], )

            trainer.save_img(src_img, train_cfg.resume_milestone, tag="test-before-source")
            trainer.save_img(pred_model1, train_cfg.resume_milestone, tag="test-before-model1")
            trainer.save_img(pred_model2, train_cfg.resume_milestone, tag="test-before-model2")
            trainer.save_img(pred_img, train_cfg.resume_milestone, tag="test-before-translation")

            torch.cuda.empty_cache()
    trainer.train()
    pass


class Trainer(object):
    def __init__(
            self,
            model1,
            model2,
            net_G_A,
            net_G_B,
            net_D_A,
            net_D_B,
            data_loader,
            train_batch_size=16,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            trans_net_lr=1e-5,
            train_wd=1e-4,
            train_num_steps=100000,
            save_every=1000,
            sample_every=1000,
            num_samples=25,
            results_folder='./results',
            amp=False,
            fp16=False,
            split_batches=True,
            log_freq=20,
            resume_milestone=0,
            cfg={},
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no',
            kwargs_handlers=[ddp_handler],
        )

        self.accelerator.native_amp = amp

        self.model1 = model1
        self.model2 = model2
        self.net_G_A = net_G_A
        self.net_G_B = net_G_B
        self.net_D_A = net_D_A
        self.net_D_B = net_D_B

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_every = save_every
        self.sample_every = sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.log_freq = log_freq

        self.train_num_steps = train_num_steps
        self.image_size = model1.image_size
        self.cfg = cfg

        # dataset and dataloader
        dl = self.accelerator.prepare(data_loader)
        self.dl = cycle(dl)

        # optimizer
        self.opt_d1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model1.parameters()),
                                    lr=train_lr, weight_decay=train_wd)
        self.opt_d2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model2.parameters()),
                                    lr=train_lr, weight_decay=train_wd)
        self.opt_G = torch.optim.Adam(filter(lambda p: p.requires_grad, list(net_G_A.parameters()) + list(net_G_B.parameters())),
                                    lr=trans_net_lr, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(filter(lambda p: p.requires_grad, list(net_D_A.parameters()) + list(net_D_B.parameters())),
                                    lr=trans_net_lr, betas=(0.5, 0.999))
        lr_lambda = lambda iter: max((1 - iter / train_num_steps) ** 0.96, cfg.trainer.min_lr)
        self.lr_scheduler_d1 = torch.optim.lr_scheduler.LambdaLR(self.opt_d1, lr_lambda=lr_lambda)
        self.lr_scheduler_d2 = torch.optim.lr_scheduler.LambdaLR(self.opt_d2, lr_lambda=lr_lambda)
        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True, parents=True)
            self.ema_d1 = EMA(model1, ema_model=None, beta=0.999,
                           update_after_step=cfg.trainer.ema_update_after_step,
                           update_every=cfg.trainer.ema_update_every)
            self.ema_d2 = EMA(model2, ema_model=None, beta=0.999,
                           update_after_step=cfg.trainer.ema_update_after_step,
                           update_every=cfg.trainer.ema_update_every)
            self.ema_G_A = EMA(net_G_A, ema_model=None, beta=0.999,
                           update_after_step=cfg.trainer.ema_update_after_step,
                           update_every=cfg.trainer.ema_update_every)
            self.ema_G_B = EMA(net_G_B, ema_model=None, beta=0.999,
                           update_after_step=cfg.trainer.ema_update_after_step,
                           update_every=cfg.trainer.ema_update_every)

        self.criterionGAN = GANLoss("lsgan").to(self.accelerator.device)
        self.tv_loss = TVLoss().to(self.accelerator.device)
        # self.vgg_loss = VGGLoss().to(self.accelerator.device)
        self.perceptual_loss = LPIPS().eval().to(self.accelerator.device)
        self.fake_C_S_buffer = ReplayBuffer(max_size=int(15 * cfg.data.batch_size))
        self.fake_C_T_buffer = ReplayBuffer(max_size=int(15 * cfg.data.batch_size))

        # step counter state

        self.step = 0
        self._model1_unwrapped = model1
        self._model2_unwrapped = model2

        # prepare model, dataloader, optimizer with accelerator

        self.model1, self.opt_d1, self.lr_scheduler_d1, self.model2, self.opt_d2, self.lr_scheduler_d2, self.net_G_A, self.net_G_B, self.opt_G, self.net_D_A, self.net_D_B, self.opt_D= \
            self.accelerator.prepare(self.model1, self.opt_d1, self.lr_scheduler_d1, self.model2, self.opt_d2, self.lr_scheduler_d2, self.net_G_A, self.net_G_B, self.opt_G, self.net_D_A, self.net_D_B, self.opt_D)
        self.logger = create_logger(root_dir=results_folder)
        self.logger.info(cfg)
        self.writer = SummaryWriter(results_folder)
        self.results_folder = Path(results_folder)
        resume_file = str(self.results_folder / f'model-{resume_milestone}.pt')
        if os.path.isfile(resume_file):
            self.load(resume_milestone)
        
        if self.step == 0 and cfg.trainer.ckpt_path1 is not None:
            self.init_from_ckpt1(cfg.trainer.ckpt_path1)
            self.init_from_ckpt2(cfg.trainer.ckpt_path2)

    def init_from_ckpt1(self, path):
        data = safe_torch_load(path,
                          map_location=lambda storage, loc: storage)
        model = self.accelerator.unwrap_model(self.model1)
        if self.cfg.trainer.ft_use_ema:
            sd = data['ema']
            new_sd = {}
            for k in sd.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]  # remove ema_model.
                    new_sd[new_k] = sd[k]
            sd = new_sd
            model.load_state_dict(sd)
        else:
            model.load_state_dict(data['model'])
            
        if 'scale_factor' in data['model']:
            model.scale_factor = data['model']['scale_factor']

    def init_from_ckpt2(self, path):
        data = safe_torch_load(path,
                          map_location=lambda storage, loc: storage)
        model = self.accelerator.unwrap_model(self.model2)
        if self.cfg.trainer.ft_use_ema:
            sd = data['ema']
            new_sd = {}
            for k in sd.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]  # remove ema_model.
                    new_sd[new_k] = sd[k]
            sd = new_sd
            model.load_state_dict(sd)
        else:
            model.load_state_dict(data['model'])
            
        if 'scale_factor' in data['model']:
            model.scale_factor = data['model']['scale_factor']

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model1': self.accelerator.get_state_dict(self.model1),
            'model2': self.accelerator.get_state_dict(self.model2),
            'net_G_A': self.accelerator.get_state_dict(self.net_G_A),
            'net_G_B': self.accelerator.get_state_dict(self.net_G_B),
            'net_D_A': self.accelerator.get_state_dict(self.net_D_A),
            'net_D_B': self.accelerator.get_state_dict(self.net_D_B),
            'opt_d1': self.opt_d1.state_dict(),
            'opt_d2': self.opt_d2.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict(),
            'lr_scheduler_d1': self.lr_scheduler_d1.state_dict(),
            'lr_scheduler_d2': self.lr_scheduler_d2.state_dict(),
            'ema_d1': self.ema_d1.state_dict(),
            'ema_d2': self.ema_d2.state_dict(),
            'ema_G_A': self.ema_G_A.state_dict(),
            'ema_G_B': self.ema_G_B.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator

        data = safe_torch_load(str(self.results_folder / f'model-{milestone}.pt'),
                          map_location=lambda storage, loc: storage)

        self._model1_unwrapped.load_state_dict(data['model1'])
        self._model2_unwrapped.load_state_dict(data['model2'])
        self.accelerator.unwrap_model(self.net_G_A).load_state_dict(data['net_G_A'])
        self.accelerator.unwrap_model(self.net_G_B).load_state_dict(data['net_G_B'])
        self.accelerator.unwrap_model(self.net_D_A).load_state_dict(data['net_D_A'])
        self.accelerator.unwrap_model(self.net_D_B).load_state_dict(data['net_D_B'])
        if 'scale_factor' in data['model1']:
            self._model1_unwrapped.scale_factor = data['model1']['scale_factor']
            self._model2_unwrapped.scale_factor = data['model2']['scale_factor']

        self.step = data['step']
        self.opt_d1.load_state_dict(data['opt_d1'])
        self.opt_d2.load_state_dict(data['opt_d2'])
        self.opt_G.load_state_dict(data['opt_G'])
        self.opt_D.load_state_dict(data['opt_D'])
        self.lr_scheduler_d1.load_state_dict(data['lr_scheduler_d1'])
        self.lr_scheduler_d2.load_state_dict(data['lr_scheduler_d2'])
        if self.accelerator.is_main_process:
            self.ema_d1.load_state_dict(data['ema_d1'])
            self.ema_d2.load_state_dict(data['ema_d2'])
            self.ema_G_A.load_state_dict(data['ema_G_A'])
            self.ema_G_B.load_state_dict(data['ema_G_B'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def mcl_fake(self, out1, out2, temperature, distributed=False):
        if distributed:
            out1 = torch.cat(GatherLayer.apply(out1), dim=0)
            out2 = torch.cat(GatherLayer.apply(out2), dim=0)
        N = out1.size(0)

        _out = [out1, out2]
        outputs = torch.cat(_out, dim=0)
        sim_matrix = outputs @ outputs.t()
        sim_matrix = sim_matrix / temperature
        sim_matrix.fill_diagonal_(-5e4)

        mask = torch.zeros_like(sim_matrix)
        mask[N:, N:] = 1
        mask.fill_diagonal_(0)

        sim_matrix = sim_matrix[N:]
        mask = mask[N:]
        mask = mask / mask.sum(1, keepdim=True)

        lsm = F.log_softmax(sim_matrix, dim=1)
        lsm = lsm * mask
        d_loss = -lsm.sum(1).mean()
        return d_loss

    def get_latent_space(self, x, tag=None):
        if "src" in tag:
            z = self._model1_unwrapped.first_stage_model.encode(x)
            z = self._model1_unwrapped.get_first_stage_encoding(z)
            z = self._model1_unwrapped.scale_factor * z
        elif "trg" in tag:
            z = self._model2_unwrapped.first_stage_model.encode(x)
            z = self._model2_unwrapped.get_first_stage_encoding(z)
            z = self._model2_unwrapped.scale_factor * z
        return z

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def cycle_train_C_disc(self, batch, ga_ind):
        split = "train"
        if ga_ind == 0:
            x_s = batch["src_img"]
            x_s = self.get_latent_space(x_s, tag="src_img")

            eps = self._model1_unwrapped.eps
            t = torch.rand(x_s.shape[0], device=x_s.device) * (1. - eps) + eps
            noise = torch.randn_like(x_s)

            C_S = -1 * x_s
            xs_noisy = self._model1_unwrapped.q_sample(x_start=x_s, noise=noise, t=t, C=C_S)
            pred_C_S, pred_noise_C_S = self._model1_unwrapped.model(xs_noisy, t)

            with torch.autograd.set_detect_anomaly(True):
                input_C_S = pred_C_S
                fake_C_T = self.net_G_A(input_C_S, t)
                recon_C_S = self.net_G_B(fake_C_T, t)
                idt_C_S = self.net_G_B(input_C_S, t)
            
                x_t = batch["trg_img"]
                x_t = self.get_latent_space(x_t, tag="trg_img")
                C_T = -1 * x_t
                noise2 = torch.randn_like(x_t)
                xt_noisy = self._model2_unwrapped.q_sample(x_start=x_t, noise=noise2, t=t, C=C_T)
                pred_C_T, pred_noise_C_T = self._model2_unwrapped.model(xt_noisy, t)

                input_C_T = pred_C_T
                fake_C_S = self.net_G_B(input_C_T, t)
                recon_C_T = self.net_G_A(fake_C_S, t)
                idt_C_T = self.net_G_A(input_C_T, t)

                loss_ldm = self.cfg.trainer.ft_weight * (F.mse_loss(pred_C_S, C_S) + F.mse_loss(pred_C_T, C_T) + F.mse_loss(pred_noise_C_S, noise) + F.mse_loss(pred_noise_C_T, noise2))
                loss_idt = F.l1_loss(idt_C_S, input_C_S) * self.cfg.trainer.idt_weight * self.cfg.trainer.cycle_weight + F.l1_loss(idt_C_T, input_C_T) * self.cfg.trainer.idt_weight * self.cfg.trainer.cycle_weight
                loss_cycle_ABA = F.l1_loss(recon_C_S, input_C_S) * self.cfg.trainer.cycle_weight
                loss_cycle_BAB = F.l1_loss(recon_C_T, input_C_T) * self.cfg.trainer.cycle_weight

                loss_G_adv_A = self.criterionGAN(self.net_D_A(fake_C_T), True)
                loss_G_adv_B = self.criterionGAN(self.net_D_B(fake_C_S), True)

                loss_perceptual = (self.perceptual_loss(input_C_S, recon_C_S).mean([1, 2, 3]) + self.perceptual_loss(input_C_T, recon_C_T).mean([1, 2, 3])).mean() * self.cfg.trainer.perceptual_weight

                loss_ddm = loss_ldm
                loss_gen_toal =  loss_idt + loss_G_adv_B + loss_G_adv_A + loss_cycle_ABA + loss_cycle_BAB + loss_perceptual + loss_ldm #+ loss_tv_A + loss_tv_B

                loss_dict = {"{}/loss_gen_toal".format(split): loss_gen_toal.detach(),
                   "{}/loss_idt".format(split): loss_idt.detach(),
                   "{}/loss_G_adv_A".format(split): loss_G_adv_A.detach(),
                   "{}/loss_G_adv_B".format(split): loss_G_adv_B.detach(),
                   "{}/loss_cycle_ABA".format(split): loss_cycle_ABA.detach(),
                   "{}/loss_cycle_BAB".format(split): loss_cycle_BAB.detach(),
                   "{}/loss_ldm".format(split): loss_ldm.detach(),
                   "{}/loss_perceptual".format(split): loss_perceptual.detach()
                   }
            
                return loss_gen_toal, loss_ddm, loss_dict

        elif ga_ind == 1:
            x_s = batch["src_img"]
            x_s = self.get_latent_space(x_s, tag="src_img")

            eps = self._model1_unwrapped.eps
            t = torch.rand(x_s.shape[0], device=x_s.device) * (1. - eps) + eps
            noise = torch.randn_like(x_s)

            C_S = -1 * x_s
            xs_noisy = self._model1_unwrapped.q_sample(x_start=x_s, noise=noise, t=t, C=C_S)
            pred_C_S, pred_noise_C_S = self._model1_unwrapped.model(xs_noisy, t)

            input_C_S = pred_C_S
            fake_C_T = self.net_G_A(input_C_S, t)
            fake_C_T = self.fake_C_T_buffer.push_and_pop(fake_C_T)
            
            x_t = batch["trg_img"]
            x_t = self.get_latent_space(x_t, tag="trg_img")
            C_T = -1 * x_t
            noise2 = torch.randn_like(x_t)
            xt_noisy = self._model2_unwrapped.q_sample(x_start=x_t, noise=noise2, t=t, C=C_T)
            pred_C_T, pred_noise_C_T = self._model2_unwrapped.model(xt_noisy, t)

            input_C_T = pred_C_T
            fake_C_S = self.net_G_B(input_C_T, t)
            fake_C_S = self.fake_C_S_buffer.push_and_pop(fake_C_S)
            
            pred_fake_C_S = self.net_D_B(fake_C_S.detach())
            pred_fake_C_T = self.net_D_A(fake_C_T.detach())
            pred_real_x_s = self.net_D_B(input_C_S.detach())
            pred_real_x_t = self.net_D_A(input_C_T.detach())

            loss_D_B_fake = self.criterionGAN(pred_fake_C_S, False)
            loss_D_B_real = self.criterionGAN(pred_real_x_s, True)
            loss_D_A_fake = self.criterionGAN(pred_fake_C_T, False)
            loss_D_A_real = self.criterionGAN(pred_real_x_t, True)

            mcl_fake_C_S = F.normalize(pred_fake_C_S.view(-1, pred_fake_C_S.shape[-1]))
            mcl_real_C_S = F.normalize(pred_real_x_s.view(-1, pred_fake_C_S.shape[-1]))
            loss_mcl_B = self.mcl_fake(mcl_fake_C_S, mcl_real_C_S, temperature=self.cfg.trainer.temp) * self.cfg.trainer.mcl_weight
            mcl_fake_C_T = F.normalize(pred_fake_C_T.view(-1, pred_fake_C_S.shape[-1]))
            mcl_real_C_T = F.normalize(pred_real_x_t.view(-1, pred_fake_C_S.shape[-1]))
            loss_mcl_A = self.mcl_fake(mcl_fake_C_T, mcl_real_C_T, temperature=self.cfg.trainer.temp) * self.cfg.trainer.mcl_weight
            loss_D_A = (loss_D_A_fake + loss_D_A_real) * 0.5
            loss_D_B = (loss_D_B_fake + loss_D_B_real) * 0.5

            loss_ldm = self.cfg.trainer.ft_weight * (F.mse_loss(pred_C_S, C_S) + F.mse_loss(pred_C_T, C_T) + F.mse_loss(pred_noise_C_S, noise) + F.mse_loss(pred_noise_C_T, noise2))
            loss_dis_total = loss_D_A + loss_D_B + loss_mcl_A + loss_mcl_B #+ loss_ldm
            loss_ddm = loss_ldm

            loss_dict = {
                   "{}/loss_D_A".format(split): loss_D_A.detach(),
                   "{}/loss_D_B".format(split): loss_D_B.detach(),
                   "{}/loss_mcl_A".format(split): loss_mcl_A.detach(),
                   "{}/loss_mcl_B".format(split): loss_mcl_B.detach(),
                   "{}/loss_ldm_D".format(split): loss_ldm.detach(),
                   "{}/loss_dis_total".format(split): loss_dis_total.detach(),
                   }

            return loss_dis_total, loss_ldm, loss_dict


    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.
                batch = next(self.dl)
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                for ga_ind in range(self.gradient_accumulate_every):

                    if self.step == 0 and ga_ind == 0:
                        self._model1_unwrapped.on_train_batch_start(batch)
                        self._model2_unwrapped.on_train_batch_start(batch)

                    with self.accelerator.autocast():

                        loss, loss_ddm, log_dict = self.cycle_train_C_disc(batch, ga_ind)

                        loss = loss / self.gradient_accumulate_every
                        loss_ddm = loss_ddm / self.gradient_accumulate_every
                        total_loss += loss.item()
                        if ga_ind == 0:
                            self.set_requires_grad([self.net_D_A, self.net_D_B], requires_grad=False)
                            self.opt_G.zero_grad()
                            self.opt_D.zero_grad()
                            self.accelerator.backward(loss, retain_graph=True) # loss_generator + loss_ddm
                            self.opt_G.step()
                            self.opt_d1.zero_grad()
                            self.opt_d2.zero_grad()
                            self.accelerator.backward(loss_ddm) # loss_ddm
                            self.opt_d1.step()
                            self.opt_d2.step()

                            loss_idt = log_dict["train/loss_idt"]
                            loss_G_adv_A = log_dict["train/loss_G_adv_A"]
                            loss_G_adv_B = log_dict["train/loss_G_adv_B"]
                            loss_cycle_ABA = log_dict["train/loss_cycle_ABA"]
                            loss_cycle_BAB = log_dict["train/loss_cycle_BAB"]
                            loss_ldm = log_dict["train/loss_ldm"]
                            loss_perceptual = log_dict["train/loss_perceptual"]
                            loss_gen_toal = log_dict["train/loss_gen_toal"]

                            log_dict['lr_d1'] = self.opt_d1.param_groups[0]['lr']
                            log_dict['lr_d2'] = self.opt_d2.param_groups[0]['lr']
                            log_dict['lr_G'] = self.opt_G.param_groups[0]['lr']
                            log_dict['lr_D'] = self.opt_D.param_groups[0]['lr']
                            describtions = dict2str(log_dict)
                            describtions = "[Train Step] {}/{}: ".format(self.step, self.train_num_steps) + describtions
                            if accelerator.is_main_process:
                                pbar.desc = describtions
                            self.set_requires_grad([self.net_D_A, self.net_D_B], requires_grad=True)
                        elif ga_ind == 1:
                            self.opt_D.zero_grad()
                            self.accelerator.backward(loss, retain_graph=True)
                            self.opt_D.step()
                            self.opt_d1.zero_grad()
                            self.opt_d2.zero_grad()
                            self.accelerator.backward(loss_ddm)
                            self.opt_d1.step()
                            self.opt_d2.step()
                            
                            loss_ldm_D = log_dict["train/loss_ldm_D"]
                            loss_D_A = log_dict["train/loss_D_A"]
                            loss_D_B = log_dict["train/loss_D_B"]
                            loss_mcl_A = log_dict["train/loss_mcl_A"]
                            loss_mcl_B = log_dict["train/loss_mcl_B"]
                            loss_dis_total = log_dict["train/loss_dis_total"]

                            log_dict['lr_d1'] = self.opt_d1.param_groups[0]['lr']
                            log_dict['lr_d2'] = self.opt_d2.param_groups[0]['lr']
                            log_dict['lr_G'] = self.opt_G.param_groups[0]['lr']
                            log_dict['lr_D'] = self.opt_D.param_groups[0]['lr']
                            describtions = dict2str(log_dict)
                            describtions = "[Train Step] {}/{}: ".format(self.step, self.train_num_steps) + describtions
                            if accelerator.is_main_process:
                                pbar.desc = describtions

                    if self.step % self.log_freq == 0:
                        log_dict['lr_d1'] = self.opt_d1.param_groups[0]['lr']
                        log_dict['lr_d2'] = self.opt_d2.param_groups[0]['lr']
                        log_dict['lr_G'] = self.opt_G.param_groups[0]['lr']
                        log_dict['lr_D'] = self.opt_D.param_groups[0]['lr']
                        describtions = dict2str(log_dict)
                        describtions = "[Train Step] {}/{}: ".format(self.step, self.train_num_steps) + describtions
                        if accelerator.is_main_process:
                            pbar.desc = describtions
                            self.logger.info(describtions)

                accelerator.clip_grad_norm_(self.model1.parameters(), 1.0)
                accelerator.clip_grad_norm_(self.model2.parameters(), 1.0)
                accelerator.clip_grad_norm_(self.net_G_A.parameters(), 1.0)
                accelerator.clip_grad_norm_(self.net_G_B.parameters(), 1.0)
                accelerator.clip_grad_norm_(self.net_D_A.parameters(), 1.0)
                accelerator.clip_grad_norm_(self.net_D_B.parameters(), 1.0)
                accelerator.wait_for_everyone()

                self.lr_scheduler_d1.step()
                self.lr_scheduler_d2.step()
                if accelerator.is_main_process:
                    self.writer.add_scalar('Other/Learning_Rate_d1', self.opt_d1.param_groups[0]['lr'], self.step)
                    self.writer.add_scalar('Other/Learning_Rate_d2', self.opt_d2.param_groups[0]['lr'], self.step)
                    self.writer.add_scalar('Other/Learning_Rate_G', self.opt_G.param_groups[0]['lr'], self.step)
                    self.writer.add_scalar('Other/Learning_Rate_D', self.opt_D.param_groups[0]['lr'], self.step)
                    self.writer.add_scalar('total_loss', total_loss, self.step)
                    self.writer.add_scalar('Generator/loss_idt', loss_idt, self.step)
                    self.writer.add_scalar('Generator/loss_G_adv_A', loss_G_adv_A, self.step)
                    self.writer.add_scalar('Generator/loss_G_adv_B', loss_G_adv_B, self.step)
                    self.writer.add_scalar('Generator/loss_cycle_ABA', loss_cycle_ABA, self.step)
                    self.writer.add_scalar('Generator/loss_cycle_BAB', loss_cycle_BAB, self.step)
                    self.writer.add_scalar('Generator/loss_ldm', loss_ldm, self.step)
                    self.writer.add_scalar('Generator/loss_perceptual', loss_perceptual, self.step)
                    self.writer.add_scalar('Generator/loss_gen_toal', loss_gen_toal, self.step)
                    self.writer.add_scalar('Discriminator/loss_D_A', loss_D_A, self.step)
                    self.writer.add_scalar('Discriminator/loss_ldm_D', loss_ldm_D, self.step)
                    self.writer.add_scalar('Discriminator/loss_D_B', loss_D_B, self.step)
                    self.writer.add_scalar('Discriminator/loss_mcl_A', loss_mcl_A, self.step)
                    self.writer.add_scalar('Discriminator/loss_mcl_B', loss_mcl_B, self.step)
                    self.writer.add_scalar('Discriminator/loss_dis_total', loss_dis_total, self.step)

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema_d1.to(device)
                    self.ema_d2.to(device)
                    self.ema_G_A.to(device)
                    self.ema_G_B.to(device)
                    self.ema_d1.update()
                    self.ema_d2.update()
                    self.ema_G_A.update()
                    self.ema_G_B.update()

                    if self.step != 0 and self.step % self.save_every == 0:
                        milestone = self.step // self.save_every
                        self.save(milestone)
                        self.model1.eval()
                        self.model2.eval()
                        self.net_G_A.eval()
                        self.net_G_B.eval()

                        with torch.no_grad():
                            src_img = batch["src_img"]

                            x_s = self.get_latent_space(src_img, tag="src_img")
                            C_S = -1 * x_s
                            c_list, noise = self._model1_unwrapped.reverse_q_sample_c_list_concat(src_img)
                            target_input = []

                            step = 1. / self._model1_unwrapped.sampling_timesteps
                            rho = 1.
                            step_indices = torch.arange(self._model1_unwrapped.sampling_timesteps, dtype=torch.float32, device=device)
                            t_steps = (self._model1_unwrapped.sigma_max ** (1 / rho) + step_indices / (self._model1_unwrapped.sampling_timesteps - 1) * (
                                    step - self._model1_unwrapped.sigma_max ** (1 / rho))) ** rho
                            t_steps = reversed(torch.cat([t_steps, torch.zeros_like(t_steps[:1])]))

                            for i in range(len(c_list[:-1])):
                                target_input.append(self.net_G_A(c_list[i], t_steps[i+1].repeat((x_s.shape[0],))))

                            target_input.append(self.net_G_A(c_list[-1], t_steps[-1].repeat((x_s.shape[0],))))
                            target_input.append(noise)
                            pred_img_trg = self._model2_unwrapped.sample_from_c_list(batch_size=src_img.shape[0], c_list=target_input)

                            pred_model1 = self._model1_unwrapped.sample(batch_size=src_img.shape[0],)
                            pred_model2 = self._model2_unwrapped.sample(batch_size=src_img.shape[0],)

                            trg_img = batch["trg_img"]
                            x_t = self.get_latent_space(trg_img, tag="trg_img")
                            c_list2, noise2 = self._model2_unwrapped.reverse_q_sample_c_list_concat(trg_img)
                            target_input = []

                            step = 1. / self._model2_unwrapped.sampling_timesteps
                            rho = 1.
                            step_indices = torch.arange(self._model2_unwrapped.sampling_timesteps, dtype=torch.float32, device=device)
                            t_steps = (self._model2_unwrapped.sigma_max ** (1 / rho) + step_indices / (self._model2_unwrapped.sampling_timesteps - 1) * (
                                    step - self._model2_unwrapped.sigma_max ** (1 / rho))) ** rho
                            t_steps = reversed(torch.cat([t_steps, torch.zeros_like(t_steps[:1])]))

                            for i in range(len(c_list2[:-1])):
                                target_input.append(self.net_G_B(c_list2[i], t_steps[i+1].repeat((x_s.shape[0],))))

                            target_input.append(self.net_G_B(c_list2[-1], t_steps[-1].repeat((x_s.shape[0],))))
                            target_input.append(noise2)
                            pred_img_src = self._model1_unwrapped.sample_from_c_list(batch_size=trg_img.shape[0], c_list=target_input)

                            self.save_img(src_img, milestone, tag="pred-source-A")
                            self.save_img(trg_img, milestone, tag="pred-source-B")
                            self.save_img(pred_model1, milestone, tag="pred-model-A")
                            self.save_img(pred_model2, milestone, tag="pred-model-B")
                            self.save_img(pred_img_trg, milestone, tag=f"pred-translation-A2B")
                            self.save_img(pred_img_src, milestone, tag=f"pred-translation-B2A")

                        self.model1.train()
                        self.model2.train()
                        self.net_G_A.train()
                        self.net_G_B.train()
                accelerator.wait_for_everyone()

                pbar.update(1)

        accelerator.print('training complete')
    
    def save_img(self, pred_img, milestone, tag=None):
        if "source" in tag:
            pred_img = (pred_img + 1) * 0.5

        nrow = 2 ** math.floor(math.log2(math.sqrt(pred_img.shape[0])))
        print(f"The image {tag} has been saved !!")
        tv.utils.save_image(pred_img, str(self.results_folder / f'sample-{milestone}-{tag}.png'), nrow=nrow)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass