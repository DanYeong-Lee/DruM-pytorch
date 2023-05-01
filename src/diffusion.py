from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import to_edge_mask



class DruM_SDE(nn.Module):
    def __init__(self, sigma_0=0.6, sigma_1=0.2, alpha=-0.5, lamb=5., eps=2e-3, loss_type='weighted'):
        super().__init__()
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.alpha = alpha
        self.lamb = lamb
        self.eps = eps
        self.loss_type = loss_type

        self.ssi_1 = self.sigma_square_integral(1)

    def sigma_square(self, t):
        return (1 - t) * (self.sigma_0 ** 2) + t * (self.sigma_1 ** 2)
    
    def sigma_square_integral(self, t):
        return (1 - t/2) * t * (self.sigma_0 ** 2) + ((t ** 2) / 2) * (self.sigma_1 ** 2)
    
    def u(self, t):
        return torch.exp(self.alpha * (self.ssi_1 - self.sigma_square_integral(t)))

    def u_a2b(self, a, b):
        return torch.exp(self.alpha * (self.sigma_square_integral(b) - self.sigma_square_integral(a)))
    
    def v(self, t):
        u = self.u(t)
        return (1 - u ** (-2)) / (2 * self.alpha)

    def v_a2b(self, a, b):
        u = self.u_a2b(a, b)
        return (1 - u ** (-2)) / (2 * self.alpha)
    
    def drift_coef(self, model, X, E, mask, t):
        destination_X, destination_E, _ = model(X, E.unsqueeze(-1), t.unsqueeze(-1), mask)
        sigma_square_t = self.sigma_square(t).unsqueeze(-1).unsqueeze(-1)
        v_t = self.v(t).unsqueeze(-1).unsqueeze(-1)
        u_t = self.u(t).unsqueeze(-1).unsqueeze(-1)

        X_term1 = self.alpha * sigma_square_t * X
        E_term1 = self.alpha * sigma_square_t * E

        X_term2 = sigma_square_t * (destination_X / u_t - X) / v_t
        E_term2 = sigma_square_t * (destination_E.squeeze(-1) / u_t - E) / v_t

        return X_term1 + X_term2, E_term1 + E_term2
    

    def diffusion_coef(self, t):
        return torch.sqrt(self.sigma_square(t))


    def get_noise(self, x, e, mask):
        x_eps = torch.randn_like(x)
        x_eps = x_eps * mask.unsqueeze(-1)
        e_eps = torch.randn_like(e)
        e_eps = torch.triu(e_eps, diagonal=1)
        e_eps = e_eps + e_eps.transpose(-1, -2)
        e_eps = e_eps * to_edge_mask(mask)
        return x_eps, e_eps

    
    def q_sample(self, x_T, e_T, mask, t):
        T = torch.ones_like(t)
        t0 = torch.zeros_like(t)

        zero_coef = self.v_a2b(t, T) / (self.u_a2b(t0, t) * self.v_a2b(t0, T))
        one_coef = self.v_a2b(t0, t) / (self.u_a2b(t, T) * self.v_a2b(t0, T))
        var = self.v_a2b(t, T) * self.v_a2b(t0, t) / self.v_a2b(t0, T)

        zero_coef = zero_coef.unsqueeze(-1).unsqueeze(-1)
        one_coef = one_coef.unsqueeze(-1).unsqueeze(-1)
        var = var.unsqueeze(-1).unsqueeze(-1)

        x_0, e_0 = self.get_noise(x_T, e_T, mask)
        x_mean = zero_coef * x_0 + one_coef * x_T
        e_mean = zero_coef * e_0 + one_coef * e_T

        x_eps, e_eps = self.get_noise(x_0, e_0, mask)
        x = x_mean + torch.sqrt(var) * x_eps
        e = e_mean + torch.sqrt(var) * e_eps
        
        return x, e
    
    
    def gamma(self, t):
        if self.loss_type == 'weighted':
            return self.sigma_square(t) / ((self.v(t) * self.u(t)) ** 2)
        elif self.loss_type == 'simple':
            return 1

    def training_loss(self, model, x_T, e_T, mask):
        time = torch.rand(x_T.shape[0]) * (1 - self.eps)
        time = time.to(x_T.device)

        x_t, e_t = self.q_sample(x_T, e_T, mask, time)
        pred_x, pred_e, _ = model(x_t, e_t.unsqueeze(-1), time.unsqueeze(-1), mask)

        gamma = self.gamma(time)
        loss_x = F.mse_loss(pred_x, x_T, reduction='none').mean(dim=(1, 2)) * gamma
        loss_e = F.mse_loss(pred_e.squeeze(-1), e_T, reduction='none').mean(dim=(1, 2)) * gamma

        return loss_x.mean() + self.lamb * loss_e.mean()

    @torch.no_grad()
    def euler_maruyama_step(self, model, x_t, e_t, mask, t, delta_t):
        drift_x, drift_e = self.drift_coef(model, x_t, e_t, mask, t)
        diffusion = self.diffusion_coef(t).unsqueeze(-1).unsqueeze(-1)

        w_x, w_e = self.get_noise(x_t, e_t, mask)
        x_new = x_t + drift_x * delta_t + diffusion * w_x * np.sqrt(delta_t)
        e_new = e_t + drift_e * delta_t + diffusion * w_e * np.sqrt(delta_t)

        x_new = x_new * mask.unsqueeze(-1)
        e_new = e_new * to_edge_mask(mask)

        return x_new, e_new

    @torch.no_grad()
    def euler_maruyama_sample(self, model, device, n_atoms, n_atom_types=4, n_steps=1000):
        n_samples = len(n_atoms)
        n_max_atoms = max(n_atoms)
        x = torch.randn(n_samples, n_max_atoms, n_atom_types, device=device)
        e = torch.randn(n_samples, n_max_atoms, n_max_atoms, device=device)
        mask = torch.zeros((n_samples, n_max_atoms))
        for i, n in enumerate(n_atoms):
            mask[i, :n] = 1.
        mask = mask.to(device)
        x, e = self.get_noise(x, e, mask)

        time_steps = np.linspace(0, 1 - self.eps, n_steps)
        delta_t = time_steps[1] - time_steps[0]
        for t in tqdm(time_steps):
            time_batch = torch.ones(n_samples, device=device) * t
            x, e = self.euler_maruyama_step(model, x, e, mask, time_batch, delta_t)
        
        return x, e
    
    @torch.no_grad()
    def langevin_step(self, model, x, e, mask, t, snr=0.16):
        drift_x, drift_e = self.drift_coef(model, x, e, mask, t)
        sigma_square_t = self.sigma_square(t).unsqueeze(-1).unsqueeze(-1)
        drift_x, drift_e = drift_x / sigma_square_t, drift_e / sigma_square_t

        w_x, w_e = self.get_noise(x, e, mask)
        w_x_norm = torch.norm(w_x.reshape(w_x.shape[0], -1), dim=-1)  # (n_samples,)
        w_e_norm = torch.norm(w_e.reshape(w_e.shape[0], -1), dim=-1)  # (n_samples,)

        drift_x_norm = torch.norm(drift_x.reshape(drift_x.shape[0], -1), dim=-1)  # (n_samples,)
        drift_e_norm = torch.norm(drift_e.reshape(drift_e.shape[0], -1), dim=-1)  # (n_samples,)
        
        step_size_x = (2 * (snr * w_x_norm / drift_x_norm)**2).unsqueeze(-1).unsqueeze(-1)
        step_size_e = (2 * (snr * w_e_norm / drift_e_norm)**2).unsqueeze(-1).unsqueeze(-1)

        x = x + step_size_x * drift_x + torch.sqrt(2 * step_size_x) * w_x
        e = e + step_size_e * drift_e + torch.sqrt(2 * step_size_e) * w_e

        return x, e
    
    @torch.no_grad()
    def predictor_corrector_step(self, model, x, e, mask, t, delta_t, n_lang_steps=1, snr=0.16):
        x, e = self.euler_maruyama_step(model, x, e, mask, t, delta_t)

        for i in range(n_lang_steps):
            x, e = self.langevin_step(model, x, e, mask, t, snr)

        return x, e
    
    @torch.no_grad()
    def predictor_corrector_sample(self, model, device, n_atoms, n_atom_types=4, n_steps=1000, n_lang_steps=1, snr=0.16):
        n_samples = len(n_atoms)
        n_max_atoms = max(n_atoms)
        x = torch.randn(n_samples, n_max_atoms, n_atom_types, device=device)
        e = torch.randn(n_samples, n_max_atoms, n_max_atoms, device=device)
        mask = torch.zeros((n_samples, n_max_atoms))
        for i, n in enumerate(n_atoms):
            mask[i, :n] = 1.
        mask = mask.to(device)
        x, e = self.get_noise(x, e, mask)

        time_steps = np.linspace(0, 1 - self.eps, n_steps)
        delta_t = time_steps[1] - time_steps[0]

        for t in tqdm(time_steps):
            time_batch = torch.ones(n_samples, device=device) * t
            x, e = self.predictor_corrector_step(model, x, e, mask, time_batch, delta_t, n_lang_steps, snr)

        return x, e