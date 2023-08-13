import torch

def convert_list_to_shape(vals, t, x_shape):
    vals = vals[t]
    return vals.reshape(len(t), *((1,) * (len(x_shape) - 1)))

def linear_beta_scheduler(beta_start, beta_end, timesteps, device):
    return (beta_end - beta_start) * torch.linspace(0, 1, timesteps + 1, device=device) + beta_start
