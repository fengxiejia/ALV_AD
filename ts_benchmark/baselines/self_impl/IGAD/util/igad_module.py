import torch
import torch.nn as nn
import torch.nn.functional as F


class IdempotentLoss(nn.Module):
    def __init__(self, model, idem_weight, tight_weight, loss_tight_clamp_ratio):
        super().__init__()
        self.training_model_copy = model
        self.idem_weight = idem_weight
        self.tight_weight = tight_weight
        self.loss_tight_clamp_ratio = loss_tight_clamp_ratio

        for param in self.training_model_copy.parameters():
            param.requires_grad = False

    def forward(self, input_data, output_data, training_model):
        cur_batch_size = input_data.shape[0]
        loss_rec = (
            F.mse_loss(input_data, output_data, reduction="none")
            .reshape(cur_batch_size, -1)
            .mean(dim=-1)
        )

        self.training_model_copy.load_state_dict(training_model.state_dict())

        idem_data = input_data.permute(0, 2, 1)
        freq_means_and_stds = torch.stack(self.get_freq_means_and_stds(idem_data)).unsqueeze(0)
        num_dims = len(freq_means_and_stds.shape) - 1
        freq_means_and_stds = freq_means_and_stds.repeat(
            idem_data.shape[0],
            *(1,) * num_dims,
        ).unbind(dim=1)

        z = self.get_noise(*freq_means_and_stds)
        z = z.permute(0, 2, 1)

        fz = training_model(z)
        f_z = fz.detach()

        ff_z = training_model(f_z)
        f_fz = self.training_model_copy(fz)

        loss_idem = F.l1_loss(f_fz, fz, reduction="mean")

        loss_tight = (
            -F.l1_loss(ff_z, f_z, reduction="none")
            .reshape(cur_batch_size, -1)
            .mean(dim=-1)
        )

        loss_tight_clamp = self.loss_tight_clamp_ratio * loss_rec
        loss_tight_clamp = torch.clamp(loss_tight_clamp, min=1e-6)
        loss_tight = torch.tanh(loss_tight / loss_tight_clamp) * loss_tight_clamp

        return self.idem_weight * loss_idem + self.tight_weight * loss_tight.mean()

    def get_freq_means_and_stds(self, x):
        freq = torch.fft.fft(x, dim=-1)
        real_mean = freq.real.mean(dim=0)
        real_std = freq.real.std(dim=0)
        imag_mean = freq.imag.mean(dim=0)
        imag_std = freq.imag.std(dim=0)
        return real_mean, real_std, imag_mean, imag_std

    def get_noise(self, real_mean, real_std, imag_mean, imag_std):
        real_std = torch.nan_to_num(real_std, nan=0.0).clamp_min(1e-6)
        imag_std = torch.nan_to_num(imag_std, nan=0.0).clamp_min(1e-6)
        freq_real = torch.normal(real_mean, real_std)
        freq_imag = torch.normal(imag_mean, imag_std)
        freq = freq_real + 1j * freq_imag
        noise = torch.fft.ifft(freq, dim=-1)
        return noise.real
