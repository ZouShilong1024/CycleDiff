import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from .utils import default, identity, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one, construct_class_by_name
from tqdm.auto import tqdm
from einops import rearrange, reduce
from functools import partial
from collections import namedtuple
from random import random, randint, sample, choice
from .encoder_decoder import DiagonalGaussianDistribution
import random
from taming.modules.losses.vqperceptual import *
from .augment import AugmentPipe
from .loss import *

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DDPM(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        sampling_timesteps = None,
        loss_type = 'l2',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        clip_x_start=True,
        input_keys=['image'],
        start_dist='normal',
        sample_type='naive',
        perceptual_weight=1.,
        use_l1=False,
        **kwargs
    ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        only_model = kwargs.pop("only_model", False)
        cfg = kwargs.pop("cfg", None)
        super().__init__()
        # assert not (type(self) == DDPM and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.input_keys = input_keys
        self.cfg = cfg
        self.scale_input = self.cfg.get('scale_input', 1)
        self.register_buffer('eps', torch.tensor(cfg.get('eps', 1e-4) if cfg is not None else 1e-4))
        self.sigma_min = cfg.get('sigma_min', 1e-2) if cfg is not None else 1e-2
        self.sigma_max = cfg.get('sigma_max', 1) if cfg is not None else 1
        self.weighting_loss = cfg.get("weighting_loss", False) if cfg is not None else False
        if self.weighting_loss:
            print('#### WEIGHTING LOSS ####')

        self.clip_x_start = clip_x_start
        self.image_size = image_size
        self.objective = objective
        self.start_dist = start_dist
        assert start_dist in ['normal', 'uniform']

        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, 10)

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        loss_main_cfg_default = {'class_name': 'ddm.loss.MSE_Loss'}
        loss_vlb_cfg_default = {'class_name': 'ddm.loss.MAE_Loss'}
        loss_main_cfg = cfg.get('loss_main', loss_main_cfg_default)
        loss_vlb_cfg = cfg.get('loss_vlb', loss_vlb_cfg_default)
        self.loss_main_func = construct_class_by_name(**loss_main_cfg)
        self.loss_vlb_func = construct_class_by_name(**loss_vlb_cfg)
        self.use_l1 = use_l1

        self.perceptual_weight = perceptual_weight
        if self.perceptual_weight > 0:
            self.perceptual_loss = LPIPS().eval()

        self.use_augment = self.cfg.get('use_augment', False)
        if self.use_augment:
            self.augment = AugmentPipe(p=0.12, xflip=1e8, yflip=1, scale=1, rotate_frac=1,
                                       aniso=1, translate_frac=1)
            print('### use augment ###\n')

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model)

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False, use_ema=False):
        sd = torch.load(path, map_location="cpu")
        if 'ema' in list(sd.keys()) and use_ema:
            sd = sd['ema']
            new_sd = {}
            for k in sd.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]    # remove ema_model.
                    new_sd[new_k] = sd[k]
            sd = new_sd
        else:
            if "model" in list(sd.keys()):
                sd = sd["model"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")


    def get_input(self, batch, return_first_stage_outputs=False, return_original_cond=False):
        assert 'image' in self.input_keys;
        if len(self.input_keys) > len(batch.keys()):
            x, *_ = batch.values()
        else:
            x = batch.values()
        return x

    def training_step(self, batch):
        z, *_ = self.get_input(batch)
        cond = batch['cond'] if 'cond' in batch else None
        loss, loss_dict = self(z, cond)
        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        if self.scale_input != 1:
            x = x * self.scale_input
        # continuous time, t in [0, 1]
        eps = self.eps  # smallest time step
        t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
        # t = torch.clamp(t, eps, 1)
        return self.p_losses(x, t, *args, **kwargs)

    def q_sample(self, x_start, noise, t, K, C):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x_noisy = x_start + K / 2 * time ** 2 + C * time + torch.sqrt(time) * noise
        return x_noisy

    def pred_x0_from_xt(self, xt, noise, t, K, C):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x0 = xt - K / 2 * time ** 2 - C * time - torch.sqrt(time) * noise
        return x0

    def pred_xtms_from_xt(self, xt, noise, K, C, t, s):
        # noise = noise / noise.std(dim=[1, 2, 3]).reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        s = s.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        mean = xt + K / 2 * s ** 2 - K * time * s - C * s - s / torch.sqrt(time) * noise
        epsilon = torch.randn_like(mean, device=xt.device)
        sigma = torch.sqrt(s * (time - s) / time)
        xtms = mean + sigma * epsilon
        return xtms

    def p_losses(self, x_start, t, *args, **kwargs):
        if self.start_dist == 'normal':
            noise = torch.randn_like(x_start)
        elif self.start_dist == 'uniform':
            noise = 2 * torch.rand_like(x_start) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        if self.use_augment:
            x_start, aug_label = self.augment(x_start)
            kwargs['augment_labels'] = aug_label
        K = torch.randn_like(x_start).clamp_(-1., 1.)
        C = -1 * x_start - K / 2             # U(t) = K/2 * t**2 + Ct, U(1) = -x0
        x_noisy = self.q_sample(x_start=x_start, noise=noise, t=t, K=K, C=C)  # (b, c, h, w)
        pred = self.model(x_noisy, t, *args, **kwargs)
        theta_pred, noise_pred = pred
        K_pred, C_pred = theta_pred.chunk(2, dim=1)
        x_rec = self.pred_x0_from_xt(x_noisy, noise_pred, t=t, K=K_pred, C=C_pred)
        loss_dict = {}
        prefix = 'train'
        target1 = torch.cat([K, C], dim=1)
        target2 = noise
        target3 = x_start
        loss_simple = 0.
        loss_vlb = 0.
        # use l1 + l2
        if self.weighting_loss:
            simple_weight1 = 1 / t
            # simple_weight2 = (t ** 2 - t + 1) / (1 - t + self.eps) ** 2  # eps prevents div 0
            simple_weight2 = 1 / (1 - t + self.eps)  # eps prevents div 0
        else:
            simple_weight1 = 1
            simple_weight2 = 1

        loss_simple += simple_weight1 * self.loss_main_func(theta_pred, target1) + \
                       simple_weight2 * self.loss_main_func(noise_pred, target2)
        if self.use_l1:
            loss_simple += simple_weight1 * (theta_pred - target1).abs().mean([1, 2, 3]) + \
                           simple_weight2 * (noise_pred - target2).abs().mean([1, 2, 3])
            loss_simple = loss_simple / 2
        # rec_weight = 2 * (1 - t.reshape(C.shape[0], 1)) ** 2
        rec_weight = 1 - t.reshape(C.shape[0], 1)
        loss_simple = loss_simple.mean()
        loss_dict.update({f'{prefix}/loss_simple': loss_simple})

        # loss_consist = torch.abs(x_noisy - self.q_sample(x_start=x_start, noise=noise_pred, t=t, C=C_pred)).mean([1, 2, 3])
        loss_vlb += self.loss_vlb_func(x_rec, target3) * rec_weight ** 2
        # loss_vlb += loss_consist
        if self.perceptual_weight > 0.:
            loss_vlb += self.perceptual_loss(x_rec, target3).mean([1, 2, 3]) * rec_weight ** 2
        loss_vlb = loss_vlb.mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss = loss_simple + loss_vlb
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    @torch.no_grad()
    def sample(self, batch_size=16, up_scale=1, cond=None, denoise=True):
        image_size, channels = self.image_size, self.channels
        if cond is not None:
            batch_size = cond.shape[0]
        self.sample_type = self.cfg.get('sample_type', 'euler')
        if self.sample_type == 'euler':
            return self.sample_fn((batch_size, channels, image_size[0], image_size[1]),
                                  up_scale=up_scale, unnormalize=True, cond=cond, denoise=denoise)
        else:
            return self.sample_fn_2order((batch_size, channels, image_size[0], image_size[1]),
                                         up_scale=up_scale, unnormalize=True, cond=cond, denoise=denoise)

    @torch.no_grad()
    def sample_fn(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False):
        batch, device, sampling_timesteps = shape[0], self.eps.device, self.sampling_timesteps
        step = 1. / self.sampling_timesteps
        time_steps = torch.tensor([step], device=device).repeat(self.sampling_timesteps)
        if denoise:
            eps = self.eps
            time_steps = torch.cat((time_steps[:-1], torch.tensor([time_steps[-1] - eps], device=device), \
                                    torch.tensor([eps], device=device)), dim=0)

        if self.start_dist == 'normal':
            img = torch.randn(shape, device=device)
        elif self.start_dist == 'uniform':
            img = 2 * torch.rand(shape, device=device) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        img = F.interpolate(img, scale_factor=up_scale, mode='bilinear', align_corners=True) * self.sigma_max
        cur_time = torch.ones((batch,), device=device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((batch,), time_step, device=device)
            if i == time_steps.shape[0] - 1:
                s = cur_time
            if cond is not None:
                pred = self.model(img, cur_time, cond)
            else:
                pred = self.model(img, cur_time)
            # C, noise = pred.chunk(2, dim=1)
            theta_pred, noise = pred[:2]
            K, C = theta_pred.chunk(2, dim=1)
            K.clamp_(-1., 1.)
            img = self.pred_xtms_from_xt(img, noise, K, C, cur_time, s)
            # img = self.pred_xtms_from_xt2(img, noise, C, cur_time, s)
            cur_time = cur_time - s
        img.clamp_(-1. * self.scale_input, 1. * self.scale_input)
        if self.scale_input != 1:
            img = img / self.scale_input
        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        return img

class LatentDiffusion(DDPM):
    def __init__(self,
                 auto_encoder,
                 scale_factor=1.0,
                 scale_by_std=True,
                 scale_by_softsign=False,
                 input_keys=['image'],
                 sample_type='naive',
                 default_scale=False,
                 *args,
                 **kwargs
                 ):
        self.scale_by_std = scale_by_std
        self.scale_by_softsign = scale_by_softsign
        self.default_scale = default_scale
        # self.perceptual_weight = 0
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        only_model = kwargs.pop("only_model", False)
        super().__init__(*args, **kwargs)
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        if self.scale_by_softsign:
            self.scale_by_std = False
            print('### USING SOFTSIGN RESCALING')
        assert (self.scale_by_std and self.scale_by_softsign) is False;

        self.init_first_stage(auto_encoder)
        # self.instantiate_cond_stage(cond_stage_config)
        self.input_keys = input_keys
        self.clip_denoised = False
        assert sample_type in ['naive', 'ddim', 'dpm', ] ###  'dpm' is not availible now, suggestion 'ddim'
        self.sample_type = sample_type

        if self.cfg.get('use_disloss', False):
            loss_dis_func_default = {'class_name': 'ddm.loss.MAE_Loss'}
            loss_dis_func = self.cfg.get('loss_dis', loss_dis_func_default)
            self.loss_dis_func = construct_class_by_name(**loss_dis_func)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model)

    def init_first_stage(self, first_stage_model):
        self.first_stage_model = first_stage_model.eval()
        # self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    '''
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    '''

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        # return self.scale_factor * z.detach() + self.scale_bias
        return z.detach()

    @torch.no_grad()
    def on_train_batch_start(self, batch):
        # only for the first batch
        if self.scale_by_std and (not self.scale_by_softsign):
            if not self.default_scale:
                assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
                # set rescale weight to 1./std of encodings
                print("### USING STD-RESCALING ###")
                x, *_ = batch.values()
                encoder_posterior = self.first_stage_model.encode(x)
                z = self.get_first_stage_encoding(encoder_posterior)
                del self.scale_factor
                self.register_buffer('scale_factor', 1. / z.flatten().std())
                print(f"setting self.scale_factor to {self.scale_factor}")
                # print("### USING STD-RESCALING ###")
            else:
                print(f'### USING DEFAULT SCALE {self.scale_factor}')
        else:
            print(f'### USING SOFTSIGN SCALE !')

    @torch.no_grad()
    def get_input(self, batch, return_first_stage_outputs=False, return_original_cond=False):
        assert 'image' in self.input_keys;
        # if len(self.input_keys) > len(batch.keys()):
        #     x, cond, *_ = batch.values()
        # else:
        #     x, cond = batch.values()
        x = batch['image']
        cond = batch['cond'] if 'cond' in batch else None
        z = self.first_stage_model.encode(x)
        z = self.get_first_stage_encoding(z)
        # if self.cfg.get('use_disloss', False):
        out = [z, cond, x]
        if return_first_stage_outputs:
            xrec = self.first_stage_model.decode(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(cond)
        return out

    def training_step(self, batch):
        z, c, x, *_ = self.get_input(batch)
        if self.scale_by_softsign:
            z = F.softsign(z)
        elif self.scale_by_std:
            z = self.scale_factor * z
        # print('grad', self.scale_bias.grad)
        if self.cfg.get('use_disloss', False):
            loss, loss_dict = self(z, c, ori_input=x)
        else:
            loss, loss_dict = self(z, c)
        return loss, loss_dict


    def p_losses(self, x_start, t, *args, **kwargs):
        if self.start_dist == 'normal':
            noise = torch.randn_like(x_start)
        elif self.start_dist == 'uniform':
            noise = 2 * torch.rand_like(x_start) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        K = torch.randn_like(x_start).clamp_(-1., 1.)
        C = -1 * x_start - K / 2  # U(t) = K/2 * t**2 + Ct, U(1) = -x0
        x_noisy = self.q_sample(x_start=x_start, noise=noise, t=t, K=K, C=C)  # (b, c, h, w)
        pred = self.model(x_noisy, t, *args, **kwargs)
        theta_pred, noise_pred = pred
        K_pred, C_pred = theta_pred.chunk(2, dim=1)
        x_rec = self.pred_x0_from_xt(x_noisy, noise_pred, t=t, K=K_pred, C=C_pred)
        loss_dict = {}
        prefix = 'train'
        target1 = torch.cat([K, C], dim=1)
        target2 = noise
        target3 = x_start
        loss_simple = 0.
        loss_vlb = 0.
        if self.weighting_loss:
            simple_weight1 = 1 / t
            simple_weight2 = 1 / (1 - t + self.eps)  # eps prevents div 0
        else:
            simple_weight1 = 1
            simple_weight2 = 1
        loss_simple += simple_weight1 * self.loss_main_func(C_pred, target1) + \
                       simple_weight2 * self.loss_main_func(noise_pred, target2)
        if self.use_l1:
            loss_simple += simple_weight1 * (C_pred - target1).abs().mean([1, 2, 3]) + \
                       simple_weight2 * (noise_pred - target2).abs().mean([1, 2, 3])
            loss_simple = loss_simple / 2
        rec_weight = (1 - t.reshape(C.shape[0], 1))
        loss_simple = loss_simple.mean()
        loss_dict.update({f'{prefix}/loss_simple': loss_simple})
        loss_vlb += self.loss_vlb_func(x_rec, target3) * rec_weight
        loss_vlb = loss_vlb.mean()
        if self.cfg.get('use_disloss', False):
            with torch.no_grad():
                img_rec = self.first_stage_model.decode(x_rec / self.scale_factor)
                img_rec = torch.clamp(img_rec, min=-1., max=1.)  # B, 1, 320, 320
            loss_tmp = self.loss_dis_func((img_rec + 1)/2, (kwargs['ori_input']+1)/2) * rec_weight ** 2  # B, 1
            if self.perceptual_weight > 0.:
                loss_tmp += self.perceptual_loss(img_rec, kwargs['ori_input']).mean([1, 2, 3]) * rec_weight ** 2
            loss_distill = SpecifyGradient.apply(x_rec, loss_tmp.mean())
            loss_vlb += loss_distill.mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss = loss_simple + loss_vlb
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    @torch.no_grad()
    def sample(self, batch_size=16, up_scale=1, cond=None, mask=None, denoise=True):
        image_size, channels = self.image_size, self.channels
        if cond is not None:
            batch_size = cond.shape[0]
        down_ratio = self.first_stage_model.down_ratio
        z = self.sample_fn((batch_size, channels, image_size[0]//down_ratio, image_size[1]//down_ratio),
                           up_scale=up_scale, unnormalize=False, cond=cond, denoise=denoise)

        if self.scale_by_std:
            z = 1. / self.scale_factor * z.detach()
        elif self.scale_by_softsign:
            z = z / (1 - z.abs())
            z = z.detach()
        #print(z.shape)
        x_rec = self.first_stage_model.decode(z)
        x_rec = unnormalize_to_zero_to_one(x_rec)
        x_rec = torch.clamp(x_rec, min=0., max=1.)
        if mask is not None:
            x_rec = mask * unnormalize_to_zero_to_one(cond) + (1 - mask) * x_rec
        return x_rec

    @torch.no_grad()
    def sample_fn(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False):
        batch, device, sampling_timesteps = shape[0], self.eps.device, self.sampling_timesteps
        step = 1. / self.sampling_timesteps
        time_steps = torch.tensor([step], device=device).repeat(self.sampling_timesteps)
        if denoise:
            eps = self.eps
            time_steps = torch.cat((time_steps[:-1], torch.tensor([time_steps[-1] - eps], device=device), \
                                    torch.tensor([eps], device=device)), dim=0)
        if self.start_dist == 'normal':
            img = torch.randn(shape, device=device)
        elif self.start_dist == 'uniform':
            img = 2 * torch.rand(shape, device=device) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')
        img = F.interpolate(img, scale_factor=up_scale, mode='bilinear', align_corners=True)
        cur_time = torch.ones((batch,), device=device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((batch,), time_step, device=device)
            if i == time_steps.shape[0] - 1:
                s = cur_time
            if cond is not None:
                pred = self.model(img, cur_time, cond)
            else:
                pred = self.model(img, cur_time)
            theta_pred, noise = pred[:2]
            K, C = theta_pred.chunk(2, dim=1)
            K.clamp_(-1., 1.)
            img = self.pred_xtms_from_xt(img, noise, K, C, cur_time, s)
            cur_time = cur_time - s

        if unnormalize:
            img = unnormalize_to_zero_to_one(img)
        return img

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones(input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

if __name__ == "__main__":
    ddconfig = {'double_z': True,
                'z_channels': 4,
                'resolution': (240, 960),
                'in_channels': 3,
                'out_ch': 3,
                'ch': 128,
                'ch_mult': [1, 2, 4, 4],  # num_down = len(ch_mult)-1
                'num_res_blocks': 2,
                'attn_resolutions': [],
                'dropout': 0.0}
    lossconfig = {'disc_start': 50001,
                  'kl_weight': 0.000001,
                  'disc_weight': 0.5}
    from encoder_decoder import AutoencoderKL
    auto_encoder = AutoencoderKL(ddconfig, lossconfig, embed_dim=4,
                                 )
    from mask_cond_unet import Unet
    unet = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=4, cond_in_dim=1,)
    ldm = LatentDiffusion(auto_encoder=auto_encoder, model=unet, image_size=ddconfig['resolution'])
    image = torch.rand(1, 3, 128, 128)
    mask = torch.rand(1, 1, 128, 128)
    input = {'image': image, 'cond': mask}
    time = torch.tensor([1])
    with torch.no_grad():
        y = ldm.training_step(input)
    pass