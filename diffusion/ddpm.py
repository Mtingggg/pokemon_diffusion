import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from .utils import convert_list_to_shape

def linear_beta_scheduler(beta_start, beta_end, timesteps, device):
    return (beta_end - beta_start) * torch.linspace(0, 1, timesteps + 1, device=device) + beta_start

tensor2numpy = transforms.Compose([
     transforms.Lambda(lambda t: (t + 1) / 2),
     transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     transforms.Lambda(lambda t: t * 255.),
     transforms.Lambda(lambda t: t.numpy().astype(np.uint8))
])

class DDPM(torch.nn.Module):
    def __init__(self, beta_start: int, beta_end: int, time_step: int, device: str):
        super(DDPM, self).__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.time_step = time_step
        self.device = device
        
        self.betas = linear_beta_scheduler(beta_start, beta_end, time_step, device)
        self.alphas = 1-self.betas
        self.alphas_cumprod = torch.cumsum(self.alphas.log(), dim=0).exp() # take log to avoid float precision issue
        self.tensor2numpy = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8))
        ])

    
    def denoise(self, x, t, pred_noise):
        x_shape = x.shape
        z = torch.randn_like(x, device=self.device)
        z[t==0]=0
        noise = convert_list_to_shape(self.betas, t, x_shape).sqrt() * z
        numerator = (1 - convert_list_to_shape(self.alphas, t, x_shape))
        denominator = (1 - convert_list_to_shape(self.alphas_cumprod, t, x_shape)).sqrt()
        mean = (x - pred_noise * (numerator/ denominator))/convert_list_to_shape(self.alphas, t, x_shape).sqrt()
        
        return mean + noise
    
    @torch.no_grad()
    def sample(self, model, n_sample, img_size, context_dim, timesteps, step_size):
        # sample initial noise
        samples = torch.randn(n_sample, 3, img_size, img_size).to(self.device)
        sample_contexts = F.one_hot(torch.randint(0, (context_dim-1), (n_sample,)), context_dim).unsqueeze(1).float().to(self.device)

        sample_steps = [[] for _ in range(n_sample)]
        for i in range(timesteps, -1, -1):
            print(f'sampling timestep {i:3d}', end='\r')
            t = torch.tensor([i]*n_sample).long().to(self.device)
            eps = model(samples, t, sample_contexts) # predict noise
            samples = self.denoise(samples, t, eps)
            if i%step_size==0:
                for n in range(n_sample):
                    sample_steps[n].append(self.tensor2numpy(samples[n].detach().cpu()))
        
        return np.array(sample_steps)
