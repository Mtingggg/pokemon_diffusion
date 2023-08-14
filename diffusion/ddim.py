import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from .ddpm import DDPM

class DDIM(DDPM):
    def __init__(self, beta_start: int, beta_end: int, time_step: int, device: str):
        super(DDIM, self).__init__(beta_start, beta_end, time_step, device)    
    
    def denoise(self, x, t, t_prev, pred_noise):
        alphas_cumprod = self.alphas_cumprod[t]
        alphas_cumprod_prev = self.alphas_cumprod[t_prev]

        mean = alphas_cumprod_prev.sqrt() / alphas_cumprod.sqrt() * (x - (1 - alphas_cumprod).sqrt() * pred_noise)
        noise = (1 - alphas_cumprod_prev).sqrt() * pred_noise

        return mean + noise

    @torch.no_grad()
    def sample(self, model, n_sample, img_size, context_dim, timesteps, step_size, plot):
        # sample initial noise
        samples = torch.randn(n_sample, 3, img_size, img_size).to(self.device)
        sample_contexts = F.one_hot(torch.randint(0, (context_dim-1), (n_sample,)), context_dim).unsqueeze(1).float().to(self.device)

        sample_steps = [[] for _ in range(n_sample)]
        for i in range(timesteps, -1, -step_size):
            print(f'sampling timestep {i:3d}', end='\r')
            t = torch.tensor([i]*n_sample).long().to(self.device)
            eps = model(samples, t, sample_contexts) # predict noise
            samples = self.denoise(samples, i, max(0,i-step_size), eps)  
            for n in range(n_sample):
                sample_steps[n].append(self.tensor2numpy(samples[n].detach().cpu()))
        res = np.array(sample_steps)

        if plot:
            fig, axes = plt.subplots(res.shape[0], res.shape[1], figsize=(len(sample_steps[0])*2,n_sample*2))
            axs = axes.flatten()
            for i in range(len(res)):
                for img_idx in range(len(res[i])):
                    axs_idx = i*res.shape[1]+img_idx
                    axs[axs_idx].clear()
                    axs[axs_idx].imshow(res[i][img_idx])
                    axs[axs_idx].set_xticks([])
                    axs[axs_idx].set_yticks([])

            plt.tight_layout()
            plt.show()

        return res
