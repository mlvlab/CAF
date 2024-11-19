import copy
import functools
import os
import wandb
from PIL import Image as PILImage
import pickle
import glob
import scipy
import dnnlib

import torch as th
import torchvision.utils as vtils
from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm import tqdm
from piq import LPIPS
import numpy as np

from flow.resample import UniformSampler, ExponentialPDF, sample_t
from flow.nn import cycle


class TrainLoop:
    def __init__(
        self,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        eval_interval,
        save_interval,
        resume,
        use_fp16=False,
        weight_decay=0.0,
        schedule_sampler='uniform',
        loss_norm='l2',
        num_classes=10,
        data_name="",
        ref_path="",
        total_training_steps=1000000,
        clip_grad_norm=False,
        clip_grad_norm_value=1.,
    ):
        
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = ema_rate
        self.resume = resume
        self.use_fp16 = use_fp16
        self.schedule_sampler = schedule_sampler
        self.num_classes = num_classes
        self.data_name = data_name
        self.ref_path = ref_path
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_norm_value = clip_grad_norm_value
        if self.schedule_sampler == 'uniform':
            self.sampler = UniformSampler(diffusion.num_timesteps)
        elif self.schedule_sampler == 'exponential':
            self.sampler = ExponentialPDF(a=0, b=1, name='ExponentialPDF')
        else:
            raise ValueError(f"Invalid schedule sampler: {self.schedule_sampler}")
        self.weight_decay = weight_decay
        self.loss_norm = loss_norm
        if 'lpips' in self.loss_norm:
            print('Using LPIPS loss...')
            self.loss_lpips = LPIPS(replace_pooling=True, reduction="none")
            self.loss_lpips = th.compile(self.loss_lpips)
        else:
            self.loss_lpips = None
        
        self.accelerator = Accelerator(
            split_batches = False,
            mixed_precision = 'fp16' if self.use_fp16 else 'no',
            even_batches = True,
            #gradient_accumulation_steps=gradient_accumulation_stepzs,
        )
        
        # Adjust total training steps, log interval, eval interval, and save interval
        if microbatch != -1:
            print(f'Global batch size: {batch_size}')
            print(f'Microbatch size: {microbatch}')
            print(f'Gradient accumulation steps: {batch_size // microbatch}')
            print(f'Adjusting total training steps according to gradient accumulation steps...')
            div = batch_size // microbatch
            self.total_training_steps = int(total_training_steps // div)
            print(f'Total training steps: {self.total_training_steps}')
        else:
            print(f'No microbatching, using total training steps: {total_training_steps}')
            print(f'Gradient accumulation steps: 1')
            self.total_training_steps = total_training_steps
            print(f'Total training steps: {self.total_training_steps}')

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        print(f'Log interval: {self.log_interval}')
        print(f'Eval interval: {self.eval_interval}')
        print(f'Save interval: {self.save_interval}')

        self.fid = 100
        if self.data_name.lower() == 'cifar10':
            if self.accelerator.is_main_process:
                print('Loading Inception-v3 model...')
                detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
                self.detector_kwargs = dict(return_features=True)
                self.feature_dim = 2048
                with dnnlib.util.open_url(detector_url, verbose=(0 == 0)) as f:
                    self.detector_net = pickle.load(f).to(self.accelerator.device)
                with dnnlib.util.open_url(self.ref_path) as f:
                    ref = dict(np.load(f))
                self.mu_ref = ref['mu']
                self.sigma_ref = ref['sigma']
        else:
            import tensorflow.compat.v1 as tf
            from flow.evaluator import Evaluator
            
            if self.accelerator.is_main_process:
                config = tf.ConfigProto(
                    allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
                )
                config.gpu_options.allow_growth = True
                config.gpu_options.per_process_gpu_memory_fraction = 0.1
                self.evaluator = Evaluator(tf.Session(config=config), batch_size=100)
                self.ref_acts = self.evaluator.read_activations(self.ref_path)
                self.ref_stats, self.ref_stats_spatial = self.evaluator.read_statistics(self.ref_path, self.ref_acts)
            
            th.cuda.empty_cache()
            tf.reset_default_graph()

    # Load Evaluation Metric Statistics
    def calculate_inception_stats(self, data_name, image_path, num_samples=50000, batch_size=100, device=th.device('cuda')):
        if data_name.lower() == 'cifar10':
            mu = th.zeros([self.feature_dim], dtype=th.float64, device=device)
            sigma = th.zeros([self.feature_dim, self.feature_dim], dtype=th.float64, device=device)
            files = glob.glob(os.path.join(image_path, 'sample*.npz'))
            count = 0
            for file in files:
                images = np.load(file)['arr_0']  # [0]#["samples"]
                for k in range((images.shape[0] - 1) // batch_size + 1):
                    mic_img = images[k * batch_size: (k + 1) * batch_size]
                    mic_img = th.tensor(mic_img).permute(0, 3, 1, 2).to(device)
                    features = self.detector_net(mic_img, **self.detector_kwargs).to(th.float64)
                    if count + mic_img.shape[0] > num_samples:
                        remaining_num_samples = num_samples - count
                    else:
                        remaining_num_samples = mic_img.shape[0]
                    mu += features[:remaining_num_samples].sum(0)
                    sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
                    count = count + remaining_num_samples
                    if count >= num_samples:
                        break
                if count >= num_samples:
                    break
            assert count == num_samples
            mu /= num_samples
            sigma -= mu.ger(mu) * num_samples
            sigma /= num_samples - 1
            mu = mu.cpu().numpy()
            sigma = sigma.cpu().numpy()
            return mu, sigma
        else:
            filenames = glob.glob(os.path.join(image_path, '*.npz'))
            imgs = []
            for file in filenames:
                try:
                    img = np.load(file)  # ['arr_0']
                    try:
                        img = img['data']
                    except:
                        img = img['arr_0']
                    imgs.append(img)
                except:
                    pass
            imgs = np.concatenate(imgs, axis=0)
            os.makedirs(os.path.join(image_path, 'single_npz'), exist_ok=True)
            np.savez(os.path.join(os.path.join(image_path, 'single_npz'), f'data'),
                     imgs)  # , labels)
            print("computing sample batch activations...")
            sample_acts = self.evaluator.read_activations(
                os.path.join(os.path.join(image_path, 'single_npz'), f'data.npz'))
            print("computing/reading sample batch statistics...")
            sample_stats, sample_stats_spatial = tuple(self.evaluator.compute_statistics(x) for x in sample_acts)
            with open(os.path.join(os.path.join(image_path, 'single_npz'), f'stats'), 'wb') as f:
                pickle.dump({'stats': sample_stats, 'stats_spatial': sample_stats_spatial}, f)
            with open(os.path.join(os.path.join(image_path, 'single_npz'), f'acts'), 'wb') as f:
                pickle.dump({'acts': sample_acts[0], 'acts_spatial': sample_acts[1]}, f)
            return sample_acts, sample_stats, sample_stats_spatial
    
    def compute_fid(self, mu, sigma, ref_mu=None, ref_sigma=None):
        if np.array(ref_mu == None).sum():
            ref_mu = self.mu_ref
            assert ref_sigma == None
            ref_sigma = self.sigma_ref
        m = np.square(mu - ref_mu).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma, ref_sigma), disp=False)
        fid = m + np.trace(sigma + ref_sigma - s * 2)
        fid = float(np.real(fid))
        return fid
    
    def update_ema(self, model, ema_model, decay):
        with th.no_grad():
            for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
                # Update EMA parameters in float32 for numerical stability
                model_param_data = model_param.data.float()
                ema_param.data.mul_(decay).add_(model_param_data, alpha=1 - decay)

    def synchronize_ema(self, ema_model):
        with th.no_grad():
            for param in ema_model.parameters():
                # Reduce EMA parameters across all processes by averaging
                # Using accelerator.sync_gradients=False to prevent gradient synchronization
                self.accelerator.reduce(param.data, reduction='mean')

    def forward_backward_velocity(self, data, latents, classes):
        self.opt.zero_grad()
        accumulate = data.shape[0] // self.microbatch

        # If accumulate is 0, skip forward_backward_velocity 
        if accumulate == 0:
            print('Accumulate is 0, skip forward_backward_acc')
            return
        
        # Forward backward for each microbatch
        for i in range(0, data.shape[0], self.microbatch):
            micro_data = data[i : i + self.microbatch]
            micro_latents = latents[i: i+self.microbatch]
            if classes != None:
                micro_classes = classes[i: i+self.microbatch]
            else:
                micro_classes = None
            if self.schedule_sampler == 'uniform':
                t = self.sampler.sample(micro_data.shape[0], self.accelerator.device)
            elif self.schedule_sampler == 'exponential':
                t = sample_t(self.sampler, micro_data.shape[0], 4).to(self.accelerator.device)
            else:
                raise ValueError(f'Invalid schedule sampler: {self.schedule_sampler}')

            with self.accelerator.autocast():
                compute_losses = functools.partial(
                    self.diffusion.velocity_training_losses,
                    self.velmodel,
                    micro_data,
                    t,
                    noise=micro_latents,
                    classes=micro_classes,
                    loss_lpips=self.loss_lpips,
                )

            losses = compute_losses()
            loss = (losses["loss"]).mean()
            loss_acc = loss / accumulate
            self.accelerator.backward(loss_acc)

        if self.clip_grad_norm and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
        self.velopt.step()
        self.velopt.zero_grad()
        self.accelerator.wait_for_everyone()

        if self.is_wandb and self.accelerator.is_main_process:
            wandb.log({'Velocity Score loss':loss.item()}, step=self.step)

    def forward_backward_acc(self, data, latents, classes):
        self.opt.zero_grad()
        accumulate = data.shape[0] // self.microbatch

        # If accumulate is 0, skip forward_backward_acc 
        if accumulate == 0:
            print('Accumulate is 0, skip forward_backward_acc')
            return
        
        # Forward backward for each microbatch
        for i in range(0, data.shape[0], self.microbatch):
            micro_data = data[i : i + self.microbatch]
            micro_latents = latents[i: i+self.microbatch]
            if classes != None:
                micro_classes = classes[i: i+self.microbatch]
            else:
                micro_classes = None

            if self.schedule_sampler == 'uniform':
                t = self.sampler.sample(micro_data.shape[0], self.accelerator.device)
            elif self.schedule_sampler == 'exponential':
                t = sample_t(self.sampler, micro_data.shape[0], 4).to(self.accelerator.device)
            else:
                raise ValueError(f'Invalid schedule sampler: {self.schedule_sampler}')

            if micro_data.shape[0] != self.microbatch:
                raise ValueError(f'Microbatch size {micro_data.shape[0]} does not match batch size {self.batch_size}')

            with self.accelerator.autocast():
                compute_losses = functools.partial(
                    self.diffusion.acceleration_training_losses,
                    self.model,
                    micro_data,
                    t,
                    noise=micro_latents,
                    velmodel=self.velmodel,
                    classes=micro_classes,
                    loss_lpips=self.loss_lpips,
                )
            losses = compute_losses()
            loss = (losses["loss"]).mean()
            loss_acc = loss / accumulate
            self.accelerator.backward(loss_acc)

        if self.clip_grad_norm and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
        self.opt.step()
        self.opt.zero_grad()
        self.accelerator.wait_for_everyone()

        if self.is_wandb and self.accelerator.is_main_process:
            wandb.log({'Acceleration Score loss':loss.item()}, step=self.step)

    def forward_backward_gan(self, data, fake_data, fake_latent, classes):
        self.opt.zero_grad()
        self.d_opt.zero_grad()
        accumulate = data.shape[0] // self.microbatch

        # If accumulate is 0, skip forward_backward_gan 
        if accumulate == 0:
            print('Accumulate is 0, skip forward_backward_gan')
            return
        
        # Forward backward for each microbatch
        for i in range(0, data.shape[0], self.microbatch):
            if data != None:
                micro_data = data[i : i + self.microbatch]
            else:
                micro_data = None

            if fake_data != None:
                micro_fake_data = fake_data[i: i+self.microbatch]
                micro_fake_latent = fake_latent[i: i+self.microbatch]
            else:
                micro_fake_data, micro_fake_latent = None, None

            if self.class_cond:
                micro_classes = classes[i: i+self.microbatch]
            else:
                micro_classes = None
            
            # Make batch size equal
            if micro_fake_data.shape[0] > micro_data.shape[0]:
                micro_fake_data = micro_fake_data[:micro_data.shape[0]]
                micro_fake_latent = micro_fake_latent[:micro_data.shape[0]]
                if self.class_cond:
                    micro_classes = micro_classes[:micro_data.shape[0]]
                else:
                    micro_classes = micro_classes
            elif micro_fake_data.shape[0] < micro_data.shape[0]:
                micro_data = micro_data[:micro_fake_data.shape[0]]

            if self.schedule_sampler == 'uniform':
                t = self.sampler.sample(micro_data.shape[0], self.accelerator.device)
            elif self.schedule_sampler == 'exponential':
                t = sample_t(self.exponential_distribution, micro_data.shape[0], 4).to(self.accelerator.device)
            else:
                raise ValueError(f'Invalid schedule sampler: {self.schedule_sampler}')

            with self.accelerator.autocast():
                compute_losses = functools.partial(
                    self.diffusion.adversarial_training_losses,
                    self.model,
                    self.velmodel,
                    micro_data,
                    t,
                    step=self.step,
                    discriminator=self.discriminator,
                    discriminator_feature_extractor=self.discriminator_feature_extractor,
                    apply_adaptive_weight=self.apply_adaptive_weight,
                    fake_data=micro_fake_data,
                    fake_latent=micro_fake_latent,
                    g_learning_period=self.g_learning_period,
                    classes=micro_classes,
                    loss_lpips=self.loss_lpips,
                )
            losses = compute_losses()

            loss = 0.
            if self.step % self.g_learning_period == 0:
                if 'd_loss' in list(losses.keys()):
                    loss += losses['d_loss'].mean()
                if 'caf_loss' in list(losses.keys()):
                    loss += losses['caf_loss'].mean()
            else:
                loss += losses['d_loss'].mean()
            loss_acc = loss / accumulate
            self.accelerator.backward(loss_acc)
            if self.clip_grad_norm and self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.accelerator.wait_for_everyone()

        if self.is_wandb and self.accelerator.is_main_process:
            if self.step % self.g_learning_period == 0:
                if 'd_loss' in list(losses.keys()):
                    wandb.log({'GAN Fake loss':losses['d_loss'].mean().item()}, step=self.step)
                if 'caf_loss' in list(losses.keys()):
                    wandb.log({'CAF loss':losses['caf_loss'].mean().item()}, step=self.step)
                if self.apply_adaptive_weight and 'd_weight' in list(losses.keys()):
                    wandb.log({'d weight':losses['d_weight'].item()}, step=self.step)
            else:
                wandb.log({'GAN Real loss':losses['d_loss'].mean().item()}, step=self.step)
    
    @th.no_grad()
    def sample_N(self, sample_dir, num_samples, batch_size, NFE):
        number = 0
        self.model.eval()
        while num_samples > number:
            latents = th.randn((batch_size, 3, self.image_size, self.image_size)).to(self.accelerator.device)
            if self.class_cond:
                y = th.randint(0, self.num_classes, (batch_size,)).to(self.accelerator.device)
            else:
                y = None
            with self.accelerator.autocast():
                sample = self.diffusion.sample(NFE, model=self.ema.ema_model, velmodel=self.velmodel, latents=latents, classes=y)
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            arr = sample.cpu().numpy()
            np.savez(os.path.join(sample_dir, 'sample_{}_{}.npz'.format(number, self.accelerator.process_index)), arr)

            number += arr.shape[0]
        self.model.train()

    @th.no_grad()
    def eval(self, NFE, metric = 'fid'):
        sample_dir = os.path.join(self.save_pth, '{}'.format(self.step))
        os.makedirs(sample_dir, exist_ok=True)
        self.sample_N(sample_dir = sample_dir, num_samples = self.eval_num_samples, batch_size = self.eval_batch_size, NFE=NFE)
        th.cuda.empty_cache()
        
        if self.data_name.lower() == 'cifar10':
            if metric == 'fid':
                mu, sigma = self.calculate_inception_stats(self.data_name,
                                                            sample_dir,
                                                            num_samples=self.eval_num_samples)
                fid = self.compute_fid(mu, sigma)
                if self.fid >= fid:
                    self.fid = fid
                    self.save()
                print(f"{self.step}-th step"
                            f" FID-{self.eval_num_samples // 1000}k: {fid}")
                if self.is_wandb:
                    wandb.log({'FID':fid}, step=self.step)
            else:
                raise ValueError
        else:
            sample_acts, sample_stats, sample_stats_spatial = self.calculate_inception_stats(self.data_name,
                                                                            sample_dir,
                                                                            num_samples=self.eval_num_samples)
            inception_score = self.evaluator.compute_inception_score(sample_acts[0])
            fid = sample_stats.frechet_distance(self.ref_stats)
            if self.fid >= fid:
                self.fid = fid
                self.save(fidbest=True)

            sfid = sample_stats_spatial.frechet_distance(self.ref_stats_spatial)
            print(f"Inception Score-{self.eval_num_samples // 1000}k:", inception_score)
            print(f"FID-{self.eval_num_samples // 1000}k:", fid)
            print(f"sFID-{self.eval_num_samples // 1000}k:", sfid)
            prec, recall = self.evaluator.compute_prec_recall(self.ref_acts[0], sample_acts[0])
            print("Precision:", prec)
            print("Recall:", recall)
            if self.is_wandb:
                wandb.log({'IS':inception_score, 'FID':fid, 'sFID':sfid, 'Precision':prec, 'Recall':recall}, step=self.step)


class CAFTrainLoopVel(TrainLoop):
    def __init__(
        self,
        *,
        is_wandb=True,
        save_pth="",
        eval_batch_size=50,
        eval_num_samples=10000,
        velmodel=None,
        image_size=32,
        class_cond=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_pth = save_pth
        os.makedirs(self.save_pth, exist_ok=True)
        os.makedirs(os.path.join(self.save_pth, 'results'), exist_ok=True)

        self.eval_batch_size = eval_batch_size
        self.eval_num_samples = eval_num_samples
        self.velmodel = velmodel
        self.image_size = image_size
        self.class_cond = class_cond
        self.step = 0
        self.global_batch = self.batch_size 

        # Optimizer
        self.velopt = th.optim.AdamW(
            self.velmodel.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        # EMA
        if self.accelerator.is_main_process:
            self.velema = EMA(self.velmodel, beta=self.ema_rate, update_every=1)
            self.velema.to(self.accelerator.device)

        self.step = 0
        self.is_wandb = is_wandb
        if is_wandb and self.accelerator.is_main_process:
            wandb.init(project='CAF', reinit=True)
            wandb.run.name='CAF-VEL'
        
        if self.resume:
            print('loading previous checkpoints')
            self.load(os.path.join(self.save_pth, 'ldm-last.pt'))
            print('Step:', self.step)

        self.velmodel, self.velopt = self.accelerator.prepare(self.velmodel, self.velopt)
        self.data = self.accelerator.prepare(self.data)
        self.data = cycle(self.data)

    @th.no_grad()
    def eval_vel(self, NFE, metric = 'fid'):
        sample_dir = os.path.join(self.save_pth, '{}'.format(self.step))
        os.makedirs(sample_dir, exist_ok=True)
        self.sample_N_vel(sample_dir = sample_dir, num_samples = self.eval_num_samples, batch_size = self.eval_batch_size, NFE=NFE)
        th.cuda.empty_cache()
        
        if self.data_name.lower() == 'cifar10':
            if metric == 'fid':
                mu, sigma = self.calculate_inception_stats(self.data_name,
                                                            sample_dir,
                                                            num_samples=self.eval_num_samples)
                fid = self.compute_fid(mu, sigma)
                if self.fid >= fid:
                    self.fid = fid
                    self.save()
                print(f"{self.step}-th step"
                            f" FID-{self.eval_num_samples // 1000}k: {fid}")
                if self.is_wandb:
                    wandb.log({'FID':fid}, step=self.step)
            else:
                raise ValueError
        else:
            sample_acts, sample_stats, sample_stats_spatial = self.calculate_inception_stats(self.data_name,
                                                                            sample_dir,
                                                                            num_samples=self.eval_num_samples)
            inception_score = self.evaluator.compute_inception_score(sample_acts[0])
            fid = sample_stats.frechet_distance(self.ref_stats)
            if self.fid >= fid:
                self.fid = fid
                self.save()

            sfid = sample_stats_spatial.frechet_distance(self.ref_stats_spatial)
            print(f"Inception Score-{self.eval_num_samples // 1000}k:", inception_score)
            print(f"FID-{self.eval_num_samples // 1000}k:", fid)
            print(f"sFID-{self.eval_num_samples // 1000}k:", sfid)
            prec, recall = self.evaluator.compute_prec_recall(self.ref_acts[0], sample_acts[0])
            print("Precision:", prec)
            print("Recall:", recall)
            if self.is_wandb:
                wandb.log({'IS':inception_score, 'FID':fid, 'sFID':sfid, 'Precision':prec, 'Recall':recall}, step=self.step)

    @th.no_grad()
    def sample_N_vel(self, sample_dir, num_samples, batch_size, NFE):
        number = 0
        while num_samples > number:
            latents = th.randn((batch_size, 3, self.image_size, self.image_size)).to(self.accelerator.device)
            if self.class_cond:
                y = th.randint(0, self.num_classes, (batch_size,)).to(self.accelerator.device)
            else:
                y = None
            with self.accelerator.autocast():
                sample = self.sample_vel(NFE, latents, y)
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            arr = sample.cpu().numpy()
            np.savez(os.path.join(sample_dir, 'sample_{}_{}.npz'.format(number, self.accelerator.process_index)), arr)

            number += arr.shape[0]

    @th.no_grad()
    def sample_vel(self, N, latents = None, classes=None):
        self.velema.ema_model.eval()
        if latents is None:
            latents = latents
        
        z = latents.detach().clone()
        batchsize = latents.shape[0]
        t = th.zeros((batchsize,)).to(z.device)
        dt = 1/N

        v0_pred = self.velema.ema_model(z, t, None, classes).detach().clone()
        z = z.detach().clone() + v0_pred / self.diffusion.alpha * dt
        z = z.clamp(-1., 1.)
        self.velema.ema_model.train()
        return z

    def run_loop(self):
        # Fix evaluation size and latent
        fix_batch_size = 50
        fix_latents = th.randn((fix_batch_size, 3, self.image_size, self.image_size)).to(self.accelerator.device)
        if self.class_cond:
            fix_classes = th.randint(0, self.num_classes, (fix_batch_size,)).to(self.accelerator.device)
        else:
            fix_classes = None

        with tqdm(initial = self.step, total = self.total_training_steps) as pbar:
            while self.step < self.total_training_steps:
                if self.class_cond:
                    latents, data, classes = next(self.data)
                    classes = th.argmax(classes, dim = 1)
                    classes = classes.to(self.accelerator.device)
                else:
                    latents, data = next(self.data)
                    classes = None
                data = data.clamp(-1., 1.)

                self.run_step(data, latents, classes)
                
                if self.step % self.save_interval == 0:
                    self.save()

                if self.accelerator.is_main_process:
                    pbar.set_description('Training...')
                    pbar.update(1)
                
                if self.step % self.log_interval == 0 and self.accelerator.is_main_process:
                    with self.accelerator.autocast():
                        test_img = self.sample_vel(N=1, latents=fix_latents, classes=fix_classes)
                    vtils.save_image(test_img, os.path.join(self.save_pth, 'results', '{}-NFE1.png'.format(self.step)), normalize=True, scale_each=True)
                    img2 = (test_img[0].data.cpu().numpy().transpose(1,2,0)+1)/2
                    img2 = PILImage.fromarray((img2 * 255).astype(np.uint8))
                    if self.is_wandb:
                        wandb.log({'Evaluation NFE=1' :wandb.Image(img2)}, step=self.step)

                if self.step % self.eval_interval == 0 and self.accelerator.is_main_process:
                    self.eval_vel(NFE=1)

    def run_step(self, data, latents, classes):
        self.forward_backward_velocity(data, latents, classes)
        if self.accelerator.is_main_process:
            self.velema.update()
        self.step += 1

    def save(self):
        if not self.accelerator.is_local_main_process:
            return
        data = {
                'step' : self.step,
                'scaler' : self.accelerator.scaler.state_dict(),
                'velmodel' : self.accelerator.get_state_dict(self.velmodel) if self.velmodel is not None else None,
                'velema' : self.velema.ema_model.state_dict() if self.velmodel is not None else None, 
                'vel_opt' : self.velopt.state_dict() if self.velmodel is not None else None,
                }
        th.save(data, os.path.join(self.save_pth, 'ldm-{}.pt'.format(self.step)))
        th.save(data, os.path.join(self.save_pth, 'ldm-last.pt'))
    
    def load(self, pth):
        data = th.load(pth, map_location= 'cpu')
        self.step = data['step']
        self.accelerator.scaler.load_state_dict(data['scaler'])
        self.velmodel.load_state_dict(data['velmodel'])
        self.velopt.load_state_dict(data['vel_opt'])
        if self.accelerator.is_main_process:
            self.velema.ema_model.load_state_dict(data['velema'])

class CAFTrainLoopAcc(TrainLoop):
    def __init__(
        self,
        *,
        is_wandb=True,
        save_pth="",
        eval_batch_size=50,
        eval_num_samples=10000,
        vel_pth="",
        velmodel=None,
        image_size=32,
        class_cond=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_pth = save_pth
        os.makedirs(self.save_pth, exist_ok=True)
        os.makedirs(os.path.join(self.save_pth, 'results'), exist_ok=True)

        self.eval_batch_size = eval_batch_size
        self.eval_num_samples = eval_num_samples
        self.vel_pth = vel_pth
        self.velmodel = velmodel
        self.image_size = image_size
        self.class_cond = class_cond
        self.step = 0
        self.global_batch = self.batch_size 

        self.opt = th.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        print('Load velocity model!')
        vel_ckpt = th.load(os.path.join(self.vel_pth, 'ldm-last.pt'), map_location='cpu')
        self.velmodel.load_state_dict(vel_ckpt['velema'])
        self.velmodel.to(self.accelerator.device)
        self.velmodel.eval()
        self.model.load_state_dict(vel_ckpt['velema'], strict=False)

        # EMA
        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta=self.ema_rate, update_every=1)
            self.ema.to(self.accelerator.device)
            self.ema.eval()

        self.step = 0
        self.is_wandb = is_wandb
        if is_wandb and self.accelerator.is_main_process:
            wandb.init(project='CAF', reinit=True)
            wandb.run.name='CAF-ACC'
        
        if self.resume:
            print('loading previous checkpoints')
            self.load(os.path.join(self.save_pth, 'ldm-last.pt'))
            print('Step:', self.step)

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.data = self.accelerator.prepare(self.data)
        self.data = cycle(self.data)

    def run_loop(self):
        fix_batch_size = 50
        fix_latents = th.randn((fix_batch_size, 3, self.image_size, self.image_size)).to(self.accelerator.device)

        if self.class_cond:
            fix_classes = th.randint(0, self.num_classes, (fix_batch_size,)).to(self.accelerator.device)
        else:
            fix_classes = None
        
        with tqdm(initial = self.step, total = self.total_training_steps) as pbar:
            while self.step < self.total_training_steps:
                if self.class_cond:
                    latents, data, classes = next(self.data)
                    classes = th.argmax(classes, dim = 1)
                else:
                    latents, data = next(self.data)
                    classes = None
                data = data.clamp(-1., 1.)

                self.run_step(data, latents, classes)
                
                if self.step % self.save_interval == 0:
                    self.save()
                
                if self.step % self.log_interval == 0 and self.accelerator.is_main_process:
                    self.model.eval()
                    with self.accelerator.autocast():
                        test_img = self.diffusion.sample(N=1, model=self.ema.ema_model, velmodel=self.velmodel, latents=fix_latents, classes=fix_classes)
                    vtils.save_image(test_img, os.path.join(self.save_pth, 'results', '{}-N=1.png'.format(self.step)), normalize=True, scale_each=True)
                    img2 = (test_img[0].data.cpu().numpy().transpose(1,2,0)+1)/2
                    img2 = PILImage.fromarray((img2 * 255).astype(np.uint8))
                    if self.is_wandb:
                        wandb.log({'Evaluation NFE=1' :wandb.Image(img2)}, step=self.step)
                        with self.accelerator.autocast():
                            test_img = self.diffusion.sample(N=5, model=self.ema.ema_model, velmodel=self.velmodel, latents=fix_latents, classes=fix_classes)
                        img2 = (test_img[0].data.cpu().numpy().transpose(1,2,0)+1)/2
                        img2 = PILImage.fromarray((img2 * 255).astype(np.uint8))
                        wandb.log({'Evaluation NFE=5' :wandb.Image(img2)}, step=self.step)
                        vtils.save_image(test_img, os.path.join(self.save_pth, 'results', '{}-N=5.png'.format(self.step)), normalize=True, scale_each=True)
                    self.model.train()
                
                if self.step % self.log_interval == 0 and self.accelerator.is_main_process:
                    self.model.eval()
                    with self.accelerator.autocast():
                        inverted_z, pred_vel = self.diffusion.inversion(1, self.ema.ema_model, self.velmodel, data, classes, return_dict=True)
                        inverted_image = self.diffusion.sample(N=1, model=self.ema.ema_model, velmodel=self.velmodel, latents = inverted_z, classes = classes, pred_vel=pred_vel)
                    vtils.save_image(data, os.path.join(self.save_pth, 'results', '{}-data.png'.format(self.step)), normalize=True, scale_each=True)
                    vtils.save_image(inverted_image, os.path.join(self.save_pth, 'results', '{}-inversion.png'.format(self.step)), normalize=True, scale_each=True)
                    with self.accelerator.autocast():
                        inverted_z, pred_vel = self.diffusion.inversion(5, self.ema.ema_model, self.velmodel, data, classes, return_dict=True)
                        inverted_image = self.diffusion.sample(N=5, model=self.ema.ema_model, velmodel=self.velmodel, latents = inverted_z, classes = classes, pred_vel=pred_vel)
                    vtils.save_image(inverted_image, os.path.join(self.save_pth, 'results', '{}-inversion-N=5.png'.format(self.step)), normalize=True, scale_each=True)
                    self.model.train()
                self.accelerator.wait_for_everyone()

                if self.step % self.eval_interval == 0 and self.accelerator.is_main_process:
                    self.eval(NFE=1)
                
                self.accelerator.wait_for_everyone()

                if self.accelerator.is_main_process:
                    pbar.set_description('Training...')
                    pbar.update(1)
    
    def run_step(self, data, latents, classes):
        self.forward_backward_acc(data, latents, classes)
        if self.accelerator.is_main_process:
            self.ema.update()
        self.step += 1

    def save(self, fidbest=False):
        if not self.accelerator.is_local_main_process:
            return
        data = {
                'step' : self.step,
                'model' : self.accelerator.get_state_dict(self.model),
                'dae_opt' : self.opt.state_dict(),
                'ema' : self.ema.ema_model.state_dict(),
                'scaler' : self.accelerator.scaler.state_dict(),
                'velmodel' : self.accelerator.get_state_dict(self.velmodel),
                }
        if fidbest:
            th.save(data, os.path.join(self.save_pth, 'fidbest.pt'))
        else:
            th.save(data, os.path.join(self.save_pth, 'ldm-{}.pt'.format(self.step)))
            th.save(data, os.path.join(self.save_pth, 'ldm-last.pt'))
    
    def load(self, pth):
        data = th.load(pth, map_location= 'cpu')
        self.model.load_state_dict(data['model'])
        self.velmodel.load_state_dict(data['velmodel'])
        self.step = data['step']
        self.opt.load_state_dict(data['dae_opt'])
        if self.accelerator.is_main_process:
            self.ema.ema_model.load_state_dict(data['ema'])
        self.accelerator.scaler.load_state_dict(data['scaler']) 


class CAFTrainLoopGAN(TrainLoop):
    def __init__(
        self,
        *,
        latents=None,
        is_wandb=True,
        save_pth="",
        eval_batch_size=50,
        eval_num_samples=10000,
        acc_pth="",
        velmodel=None,
        d_lr=1e-3,
        discriminator=None,
        discriminator_feature_extractor=None,
        fake_cllt=None,
        g_learning_period=1,
        image_size=32,
        class_cond=False,
        apply_adaptive_weight=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latents = latents
        self.save_pth = save_pth
        os.makedirs(self.save_pth, exist_ok=True)
        os.makedirs(os.path.join(self.save_pth, 'results'), exist_ok=True)

        self.eval_batch_size = eval_batch_size
        self.eval_num_samples = eval_num_samples
        self.acc_pth = acc_pth
        self.velmodel = velmodel
        self.d_lr = d_lr
        self.discriminator = discriminator
        self.discriminator_feature_extractor = discriminator_feature_extractor
        self.fake_cllt = fake_cllt
        self.g_learning_period = g_learning_period
        self.image_size = image_size
        self.class_cond = class_cond
        self.apply_adaptive_weight = apply_adaptive_weight

        self.step = 0
        self.global_batch = self.batch_size 
        
        print('loading pre-trained veloctiy and acceleration model!')
        self.load_previous(os.path.join(self.acc_pth, 'fidbest.pt'))

        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta = self.ema_rate, update_every=1)
            self.ema.to(self.accelerator.device)

        self.opt = th.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.discriminator != None:
            self.d_opt = th.optim.AdamW(
            self.discriminator.parameters(), lr=self.d_lr, weight_decay=self.weight_decay
            )

        if self.resume:
            print('Load previous checkpoints!')
            self.load(os.path.join(self.save_pth, 'fidbest.pt'))

        self.is_wandb = is_wandb
        if is_wandb and self.accelerator.is_main_process:
            wandb.init(project='CAF', reinit=True)
            wandb.run.name='CAF-GAN'

        ## Prepare
        self.velmodel = self.velmodel.to(self.accelerator.device)
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.data, self.fake_cllt, self.latents = self.accelerator.prepare(self.data, self.fake_cllt, self.latents)
        if self.discriminator != None:
            self.discriminator, self.discriminator_feature_extractor = self.accelerator.prepare(self.discriminator, self.discriminator_feature_extractor)
            self.d_opt = self.accelerator.prepare(self.d_opt)

        self.data = cycle(self.data)
        self.fake_cllt = cycle(self.fake_cllt)

    def run_loop(self):
        fix_latents = th.randn((100, 3, self.image_size, self.image_size)).to(self.accelerator.device)
        if self.class_cond:
            fix_classes = th.randint(0, self.num_classes, (100,)).to(self.accelerator.device)
        else:
            fix_classes = None

        with tqdm(initial = self.step, total = self.total_training_steps) as pbar:
            while self.step < self.total_training_steps:
                if self.data != None:
                    data, test_class = next(self.data)
                    data = data * 2 - 1
                else:
                    data = None

                if self.fake_cllt != None:
                    if self.class_cond:
                        fake_latent, fake_data, classes = next(self.fake_cllt)
                    else:
                        fake_latent, fake_data = next(self.fake_cllt)
                        classes = None
                    fake_data = fake_data.clamp(-1, 1)
                    if self.class_cond:
                        classes = th.argmax(classes, dim = 1)
                else:
                    fake_data, fake_latent = None, None
                
                self.run_step(data, fake_data, fake_latent, classes)
                
                if self.step % self.save_interval == 0:
                    self.save()
                
                if self.step % self.log_interval == 0 and self.accelerator.is_main_process:
                    test_img = self.diffusion.sample(N=1, model=self.ema.ema_model, velmodel=self.velmodel, latents=fix_latents, classes=fix_classes)
                    vtils.save_image(test_img, os.path.join(self.save_pth, 'results', '{}-N=1.png'.format(self.step)), normalize=True, scale_each=True)
                    img2 = (test_img[0].data.cpu().numpy().transpose(1,2,0)+1)/2
                    img2 = PILImage.fromarray((img2 * 255).astype(np.uint8))
                    if self.is_wandb:
                        wandb.log({'Evaluation NFE=1' :wandb.Image(img2)}, step=self.step)
                        test_img = self.diffusion.sample(N=5, model=self.ema.ema_model, velmodel=self.velmodel, latents=fix_latents, classes=fix_classes)
                        img2 = (test_img[0].data.cpu().numpy().transpose(1,2,0)+1)/2
                        img2 = PILImage.fromarray((img2 * 255).astype(np.uint8))
                        wandb.log({'Evaluation NFE=4' :wandb.Image(img2)}, step=self.step)
                        vtils.save_image(test_img, os.path.join(self.save_pth, 'results', '{}-N=5.png'.format(self.step)), normalize=True, scale_each=True)
                self.accelerator.wait_for_everyone()

                if self.step % self.log_interval == 0 and self.accelerator.is_main_process:
                    inverted_z, pred_vel = self.diffusion.inversion(1, self.ema.ema_model, self.velmodel, data, classes, return_dict=True)
                    inverted_image = self.diffusion.sample(N=1, model=self.ema.ema_model, velmodel=self.velmodel, latents = inverted_z, classes = classes, pred_vel=pred_vel)
                    vtils.save_image(data, os.path.join(self.save_pth, 'results', '{}-data.png'.format(self.step)), normalize=True, scale_each=True)
                    vtils.save_image(inverted_image, os.path.join(self.save_pth, 'results', '{}-inversion.png'.format(self.step)), normalize=True, scale_each=True)
                    inverted_z, pred_vel = self.diffusion.inversion(5, self.ema.ema_model, self.velmodel, data, classes, return_dict=True)
                    inverted_image = self.diffusion.sample(N=5, model=self.ema.ema_model, velmodel=self.velmodel, latents = inverted_z, classes = classes, pred_vel=pred_vel)
                    vtils.save_image(inverted_image, os.path.join(self.save_pth, 'results', '{}-inversion-N=5.png'.format(self.step)), normalize=True, scale_each=True)
                
                self.accelerator.wait_for_everyone()

                if self.step % self.eval_interval == 0 and self.accelerator.is_main_process:
                   self.eval(NFE=1)
                
                self.accelerator.wait_for_everyone()

                if self.accelerator.is_main_process:
                    pbar.set_description('Training...')
                    pbar.update(1)

    
    def run_step(self, data, fake_data, fake_latent, classes):
        self.forward_backward_gan(data, fake_data, fake_latent, classes)
        if self.step % self.g_learning_period == 0:
            self.opt.step()
            self.opt.zero_grad()
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.ema.update()
        else:
            self.d_opt.step()
            self.d_opt.zero_grad()
            self.accelerator.wait_for_everyone()
        self.step += 1
    
    def save(self, fidbest=False):
        if not self.accelerator.is_local_main_process:
            return
        data = {
                'step' : self.step,
                'model' : self.accelerator.get_state_dict(self.model),
                'ema' : self.ema.ema_model.state_dict(),
                'opt' : self.opt.state_dict(),
                'velmodel' : self.accelerator.get_state_dict(self.velmodel),
                'scaler' : self.accelerator.scaler.state_dict(),
                'discrim' : self.accelerator.get_state_dict(self.discriminator),
                'd_opt' : self.d_opt.state_dict(),
                }
        if fidbest:
            th.save(data, os.path.join(self.save_pth, 'fidbest.pt'))    
        else:
            th.save(data, os.path.join(self.save_pth, 'ldm-{}.pt'.format(self.step)))
            th.save(data, os.path.join(self.save_pth, 'ldm-last.pt'))
    
    def load(self, pth):
        data = th.load(pth, map_location= 'cpu')
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        if self.accelerator.is_main_process:
            self.ema.ema_model.load_state_dict(data['ema'])
        self.opt.load_state_dict(data['opt'])
        self.velmodel.load_state_dict(data['velmodel'])
        self.accelerator.scaler.load_state_dict(data['scaler'])
        if data['discrim'] != None:
            self.discriminator.load_state_dict(data['discrim'])
        if data['d_opt'] != None:
            self.d_opt.load_state_dict(data['d_opt'])
    
    def load_previous(self, pth):
        data = th.load(pth, map_location= 'cpu')
        self.model.load_state_dict(data['ema'])
        self.velmodel.load_state_dict(data['velmodel'])