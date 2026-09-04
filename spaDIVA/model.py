from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from .compute_poe import compute_poe_gaussian

__all__ = ["VariationalAutoEncoder"]


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, w_dim):
        super().__init__()
        self.fc1 = GCNConv(input_dim, hidden_dim, add_self_loops=False)
        self.fc2 = GCNConv(hidden_dim, w_dim, add_self_loops=False)
        self.fc3 = GCNConv(hidden_dim, w_dim, add_self_loops=False)

    def forward(self, x, e, edge_w=None):
        hidden = F.elu(self.fc1(x, e, edge_weight=edge_w))
        loc = self.fc2(hidden, e, edge_weight=edge_w)
        scale = self.fc3(hidden, e, edge_weight=edge_w)
        return loc, scale


class Decoder(nn.Module):
    def __init__(self, w_dim, z_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = GCNConv(w_dim + z_dim, hidden_dim, add_self_loops=False)
        self.fc2 = GCNConv(hidden_dim, output_dim, add_self_loops=False)

    def forward(self, z, w, e, edge_w=None):
        hidden = F.elu(self.fc1(torch.concat((z, w), dim=1), e, edge_weight=edge_w))
        x_loc = self.fc2(hidden, e, edge_weight=edge_w)
        return x_loc


class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim1,
        input_dim2,
        hidden_dim1,
        hidden_dim2,
        w1_dim,
        w2_dim,
        z_dim,
        KL_weight=None,
        use_cuda=False,
    ):
        super().__init__()
        self.encoder1 = Encoder(input_dim1, hidden_dim1, w1_dim + z_dim)
        self.encoder2 = Encoder(input_dim2, hidden_dim2, w2_dim + z_dim)

        self.decoder1 = Decoder(w1_dim, z_dim, hidden_dim1, input_dim1)
        self.decoder2 = Decoder(w2_dim, z_dim, hidden_dim2, input_dim2)

        self.w1_dim, self.w2_dim, self.z_dim = w1_dim, w2_dim, z_dim
        self.KL_weight = [1, 1, 1, 1] if KL_weight is None else KL_weight

    def reparameterize(self, z_loc, z_scale, num_samples=1):
        z_samples = []
        for _ in range(num_samples):
            eps = torch.randn_like(z_loc)
            z = z_loc + eps * (torch.exp(z_scale) ** 0.5)
            z_samples.append(z)

        z_samples = torch.stack(z_samples)
        z_avg = torch.mean(z_samples, dim=0)
        return z_avg

    def forward(self, x1, x2, e, edge_weight=None, edge_index2=None, edge_weight2=None):
        edge_index1 = e
        edge_weight1 = edge_weight
        if edge_index2 is None:
            edge_index2 = edge_index1
        if edge_weight2 is None:
            edge_weight2 = edge_weight1

        loc1, scale1 = self.encoder1(x1, edge_index1, edge_weight1)
        loc2, scale2 = self.encoder2(x2, edge_index2, edge_weight2)

        w1_loc, w1_scale = loc1[:, :self.w1_dim], scale1[:, :self.w1_dim]
        w2_loc, w2_scale = loc2[:, :self.w2_dim], scale2[:, :self.w2_dim]
        z1_loc, z1_scale = loc1[:, self.w1_dim:], scale1[:, self.w1_dim:]
        z2_loc, z2_scale = loc2[:, self.w2_dim:], scale2[:, self.w2_dim:]

        w1 = self.reparameterize(w1_loc, w1_scale)
        w2 = self.reparameterize(w2_loc, w2_scale)

        z_loc, z_scale = compute_poe_gaussian(z1_loc, z1_scale, z2_loc, z2_scale)
        z = self.reparameterize(z_loc, z_scale)

        x1_loc = self.decoder1(z, w1, edge_index1, edge_weight1)
        x2_loc = self.decoder2(z, w2, edge_index2, edge_weight2)

        return x1_loc, x2_loc, z1_loc, z1_scale, z2_loc, z2_scale, w1_loc, w1_scale, w2_loc, w2_scale, z

    def loss_function(
        self,
        x1_loc,
        x2_loc,
        x1,
        x2,
        z1_loc,
        z1_scale,
        z2_loc,
        z2_scale,
        w1_loc,
        w1_scale,
        w2_loc,
        w2_scale,
        weight,
    ):
        recon_loss = 0
        recon_loss1 = F.mse_loss(x1, x1_loc, reduction="mean")
        recon_loss2 = F.mse_loss(x2, x2_loc, reduction="mean")
        recon_loss = recon_loss1 + recon_loss2

        KL_loss = 0
        KL_loss_z1 = -0.5 * torch.mean(1 + z1_scale - z1_loc.pow(2) - torch.exp(z1_scale))
        KL_loss_z2 = -0.5 * torch.mean(1 + z2_scale - z2_loc.pow(2) - torch.exp(z2_scale))
        KL_loss_w1 = -0.5 * torch.mean(1 + w1_scale - w1_loc.pow(2) - torch.exp(w1_scale))
        KL_loss_w2 = -0.5 * torch.mean(1 + w2_scale - w2_loc.pow(2) - torch.exp(w2_scale))
        KL_loss = (
            self.KL_weight[0] * KL_loss_z1
            + self.KL_weight[1] * KL_loss_z2
            + self.KL_weight[2] * KL_loss_w1
            + self.KL_weight[3] * KL_loss_w2
        )

        return recon_loss + weight * KL_loss
