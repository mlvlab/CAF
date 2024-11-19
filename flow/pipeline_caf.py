import torch as th
import torch.nn.functional as F

from flow.nn import mean_flat, append_dims
from flow.enc_dec_lib import get_xl_feature
from flow.image_datasets import mask_generator

# Loss functions
def loss_func_huber(x, y):
    data_dim = x.shape[1] * x.shape[2] * x.shape[3]
    huber_c = 0.00054 * data_dim
    loss = th.sum((x - y)**2, dim = (1, 2, 3))
    loss = th.sqrt(loss + huber_c**2) - huber_c
    return loss / data_dim

def loss_func_lpips(x, y, loss_lpips):
    return loss_lpips(x, y)

## Constant Acceleration Flow ##
class CAFDenoiser:
    def __init__(
        self,
        alpha = 1.5,
        num_timesteps=1000,
        loss_norm='l2',
    ):
        self.alpha = alpha
        self.num_timesteps = num_timesteps
        self.loss_norm = loss_norm

    def get_train_tuple(self, x_start, noise, t):
        target_vel = self.alpha * (x_start - noise)
        x_t = (1 - t**2)*noise + ((t**2))*x_start
        return target_vel, x_t

    def get_sample_timesteps(self, N, start, end, device):
        times = th.linspace(start, end, N+1).long().to(device) / (end)
        return times
    
    def denoise(self, model, x_t, sigmas, v0=None, y=None, **model_kwargs):
        model_output = model(x_t, sigmas, v0, y)
        return model_output

    def calculate_adaptive_weight(self, loss1, loss2, last_layer=None):
        loss1_grad = th.autograd.grad(loss1, last_layer, retain_graph=True)[0]
        loss2_grad = th.autograd.grad(loss2, last_layer, retain_graph=True)[0]
        d_weight = th.norm(loss1_grad) / (th.norm(loss2_grad) + 1e-4)
        d_weight = th.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight


    def get_caf_estimate(self, model, x_t, t, pred_vel, s, classes=None, **model_kwargs):
        dims = x_t.ndim
        dt = (s - t) / 1
        dt_dims = append_dims(dt, dims)
        mean_t = (s - t) / 2 + t
        mean_t_dim = append_dims(mean_t, dims)

        model_output = self.denoise(model, x_t, t, pred_vel, classes, **model_kwargs)
        one_step_estimate = x_t + pred_vel * dt_dims + mean_t_dim * model_output * dt_dims
        return one_step_estimate, model_output
    
    def get_GAN_loss(self, model, real=None, fake=None, adaptive_loss=None,
                               learn_generator=True, discriminator=None, 
                               discriminator_feature_extractor=None, 
                               apply_adaptive_weight=True,
                               step=0, init_step=0, **model_kwargs):

        if learn_generator:
            logits_fake = get_xl_feature(fake, feature_extractor=discriminator_feature_extractor,
                                                  discriminator=discriminator, **model_kwargs)
            g_loss = sum([(-l).mean() for l in logits_fake]) / len(logits_fake)
            if apply_adaptive_weight:
                d_weight = self.calculate_adaptive_weight(10*adaptive_loss.mean(), g_loss.mean(),
                                                    last_layer=model.module.output_blocks[15][0].out_layers[3].weight)
                d_weight = th.clip(d_weight, 0.00005, 10)
            else:
                d_weight = 0.00005
            discriminator_loss = d_weight * g_loss
        else:
            logits_fake, logits_real = get_xl_feature(fake.detach(), target=real.detach(),
                                                      feature_extractor=discriminator_feature_extractor,
                                                      discriminator=discriminator, step=step, **model_kwargs)
            loss_Dgen = sum([(F.relu(th.ones_like(l) + l)).mean() for l in logits_fake]) / len(logits_fake)
            loss_Dreal = sum([(F.relu(th.ones_like(l) - l)).mean() for l in logits_real]) / len(logits_real)
            discriminator_loss = loss_Dreal + loss_Dgen
            d_weight = None
        return discriminator_loss, d_weight

    # Losses
    def velocity_training_losses(
            self, 
            model, 
            x_start, 
            sigmas,
            noise=None,
            classes=None,
            loss_lpips=None,
            model_kwargs=None
        ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        terms = {}

        dims = x_start.ndim

        rescaled_t = sigmas
        rescaled_t_dims = append_dims(rescaled_t, dims)
        target_vel, x_t = self.get_train_tuple(x_start, noise, rescaled_t_dims)
        x_t = x_t + target_vel*(rescaled_t_dims - rescaled_t_dims**2)

        pred_vel = self.denoise(model, x_t, rescaled_t, None, classes)

        with th.cuda.amp.autocast(dtype=th.float32): # For numerical accuracy
            if self.loss_norm == 'l2':
                loss = mean_flat((pred_vel - target_vel) ** 2)
            elif self.loss_norm == 'lpips_huber':
                assert loss_lpips is not None
                pred_x_start = x_t - pred_vel / self.alpha * (rescaled_t_dims - 1)
                x_up = F.interpolate(x_start, size=224, mode="bilinear")
                pred_x_up = F.interpolate(pred_x_start, size=224, mode="bilinear")
                loss_huber = mean_flat(loss_func_huber(pred_vel, target_vel))
                loss_lp = mean_flat(loss_func_lpips((pred_x_up+1)/2., (x_up+1)/2., loss_lpips))
                loss = loss_huber + loss_lp
            elif self.loss_norm == 'l2_huber':
                assert loss_lpips is not None
                x_up = F.interpolate(x_start, size=224, mode="bilinear")
                pred_x_up = F.interpolate(pred_x_start, size=224, mode="bilinear")
                loss_l2 = mean_flat((pred_vel - target_vel) ** 2)
                loss_lp = mean_flat(loss_func_lpips((pred_x_up+1)/2., (x_up+1)/2., loss_lpips))
                loss = loss_l2 + loss_lp
            elif self.loss_norm == 'huber':
                loss = mean_flat(loss_func_huber(pred_vel, target_vel))
            else:
                raise ValueError

        terms["loss"] = loss
        return terms

    def acceleration_training_losses(
            self, 
            model, 
            x_start, 
            sigmas, 
            velmodel=None, 
            noise=None, 
            classes=None,
            loss_lpips=None,
            pred_vel=None,
            model_kwargs=None,
            ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            print('hi')
            noise = th.randn_like(x_start)

        terms = {}

        dims = x_start.ndim
        rescaled_t = sigmas
        rescaled_t_dims = append_dims(rescaled_t, dims)
        target_vel, x_t = self.get_train_tuple(x_start, noise, rescaled_t_dims)
        t_last = th.zeros((x_t.shape[0],)).to(x_t.device)
    
        if pred_vel is None:
            with th.no_grad():
                pred_vel = self.denoise(velmodel, noise, t_last, None, classes).detach().clone()
        
        x_t = x_t + pred_vel*(rescaled_t_dims - rescaled_t_dims**2)
        target_acc = 2*(x_start - noise) - 2*pred_vel

        pred_x_start, pred_acc = self.get_caf_estimate(model=model, 
                                            x_t=x_t, 
                                            t=rescaled_t, 
                                            pred_vel=pred_vel, 
                                            s=th.ones_like(t_last), 
                                            classes=classes, 
                                            **model_kwargs)
        
        with th.cuda.amp.autocast(dtype=th.float32): # For numerical accuracy

            if self.loss_norm == 'l2':
                loss = mean_flat((pred_acc - target_acc) ** 2)
            elif self.loss_norm == 'lpips_huber':
                assert loss_lpips is not None
                x_up = F.interpolate(x_start, size=224, mode="bilinear")
                pred_x_up = F.interpolate(pred_x_start, size=224, mode="bilinear")
                loss_huber = mean_flat(loss_func_huber(pred_acc, target_acc))
                loss_lp = mean_flat(loss_func_lpips((pred_x_up+1)/2., (x_up+1)/2., loss_lpips))
                loss = loss_huber + loss_lp
            elif self.loss_norm == 'l2_huber':
                assert loss_lpips is not None
                x_up = F.interpolate(x_start, size=224, mode="bilinear")
                pred_x_up = F.interpolate(pred_x_start, size=224, mode="bilinear")
                loss_l2 = mean_flat((pred_acc - target_acc) ** 2)
                loss_lp = mean_flat(loss_func_lpips((pred_x_up+1)/2., (x_up+1)/2., loss_lpips))
                loss = loss_l2 + loss_lp
            elif self.loss_norm =='huber':
                loss = mean_flat(loss_func_huber(pred_acc, target_acc))
            else:
                raise ValueError
        
        terms["loss"] = loss

        return terms

    def adversarial_training_losses(
        self,
        model,
        velmodel,
        x_start,
        t,
        step=0,
        discriminator=None,
        discriminator_feature_extractor=None,
        apply_adaptive_weight=False,
        fake_data=None,
        fake_latent=None,
        g_learning_period=1,
        classes=None,
        loss_lpips=None,
        model_kwargs=None,
    ):
        assert fake_data is not None and fake_latent is not None
        if model_kwargs is None:
            model_kwargs = {}
        dims = fake_data.ndim
        terms = {}

        t_dims = append_dims(t, dims)
        target_vel, x_t = self.get_train_tuple(fake_data, fake_latent, t_dims)
        t_last = th.zeros((x_t.shape[0],)).to(x_t.device)
        
        with th.no_grad():
            pred_vel = self.denoise(velmodel, fake_latent, t_last, None, classes).detach().clone()
        x_t = x_t + pred_vel * (t_dims - t_dims**2)

        pred_x_start, pred_acc = self.get_caf_estimate(model=model, 
                                                    x_t=fake_latent, 
                                                    t=t_last, 
                                                    pred_vel=pred_vel, 
                                                    s=th.ones_like(t_last), 
                                                    classes=classes, 
                                                    **model_kwargs)

        if step % g_learning_period == 0:
            terms['caf_loss'] = self.acceleration_training_losses(model=model, 
                                                                x_start=fake_data, 
                                                                sigmas=t, 
                                                                velmodel=velmodel, 
                                                                noise=fake_latent, 
                                                                classes=classes, 
                                                                loss_lpips=loss_lpips, 
                                                                pred_vel=pred_vel)["loss"]

            terms['d_loss'], d_weight = self.get_GAN_loss(model=model, 
                                                        fake=pred_x_start, 
                                                        adaptive_loss=terms['caf_loss'], 
                                                        discriminator=discriminator, 
                                                        discriminator_feature_extractor=discriminator_feature_extractor, 
                                                        apply_adaptive_weight=apply_adaptive_weight)
            terms['d_weight'] = d_weight
        
        else:
            terms['d_loss'], d_weight = self.get_GAN_loss(model=None, 
                                                        real=x_start, 
                                                        fake=pred_x_start, 
                                                        learn_generator=False, 
                                                        discriminator=discriminator, 
                                                        discriminator_feature_extractor=discriminator_feature_extractor,
                                                        )
        return terms

    @th.no_grad()
    def sample(self, N, model, velmodel, latents, classes=None, mask=None, image=None, start = 0, end = 1000, pred_vel=None, return_dict=False):
        dims = latents.ndim
        times = self.get_sample_timesteps(N, start, end, device=latents.device)
        t = th.zeros((latents.shape[0],)).to(latents.device)
        dt = (end - start) / (end * N)
        z = latents.detach().clone()

        if pred_vel is None:
            pred_vel = velmodel(z, t, None, classes).detach().clone()

        for i in range(len(times)-1):
            time = times[i]
            t_input = th.ones_like(t) * time 
            t_prime = th.ones_like(t) * time + 1/ (2 * N)
            t_prime_dim = append_dims(t_prime, dims)

            pred_acc = model(z, t_input, pred_vel, classes)
            
            z = z + pred_vel * dt + t_prime_dim * pred_acc * dt

            # Inpainting
            if mask is not None:
                z = image * mask + z * (1-mask)
        z = z.clamp(-1., 1.)

        if return_dict:
            return (z, pred_vel)
        else:
            return z

    @th.no_grad()
    def inversion(self, N, model, velmodel, latents, classes=None, mask=None, start=0, end=1000, pred_vel=None, return_dict=False):
        dims = latents.ndim
        times = self.get_sample_timesteps(N, start, end, device=latents.device)
        reverse_times = th.flip(times, dims = [0])
        t = th.ones((latents.shape[0],)).to(latents.device)
        dt = (end - start) / (end * N)
        z = latents.detach().clone()
        
        if pred_vel is None:
            pred_vel = velmodel(z, t, None, classes).detach().clone()

        for i in range(len(times)-1):
            reverse_time = reverse_times[i]
            t_input = th.ones_like(t) * reverse_time 
            t_prime = th.ones_like(t) * reverse_time - 1/ (2 * N)
            t_prime_dim = append_dims(t_prime, dims)
            
            pred_acc = model(z, t_input, pred_vel, classes)
            
            z = z - pred_vel * dt - t_prime_dim * pred_acc * dt
            
            # Inpainting
            if mask is not None:
                z = z * mask + (1-reverse_time) * th.randn_like(z) * (1-mask)

        if return_dict:
            return (z, pred_vel)
        else:
            return z

    @th.no_grad()
    def inpainting(
        self,
        N,
        model,
        velmodel,
        images,
        classes,
        mask_len_range = (15, 16),
        mask_prob_range = (0.3, 0.3),
    ):
        mask = mask_generator('box', mask_len_range = mask_len_range, mask_prob_range=mask_prob_range)(images).type(th.long)

        def replacement(x0, x1):
            x_mix = x0 * mask + x1 * (1 - mask)
            return x_mix

        images = replacement(images, 1.*th.randn_like(images))
        images_mask = replacement(images, -th.ones_like(images))

        latents = self.inversion(N, model, velmodel, images, classes, mask)
        inpainted_images = self.sample(N, model, velmodel, latents, classes, mask, images)
        
        return inpainted_images, images_mask
    