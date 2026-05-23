import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_layer import GraphLayer


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i * edge_num : (i + 1) * edge_num] += i * node_num

    return batch_edge_index.long()


def build_complete_edge_index(node_num):
    src = torch.arange(node_num).repeat_interleave(node_num)
    dst = torch.arange(node_num).repeat(node_num)
    return torch.stack([src, dst], dim=0).long()


class OutLayer(nn.Module):
    def __init__(self, in_num, layer_num, inter_num=512):
        super().__init__()
        modules = []

        for i in range(layer_num):
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x
        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)
        return out


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1):
        super().__init__()
        self.gnn = GraphLayer(
            in_channel,
            out_channel,
            inter_dim=inter_dim,
            heads=heads,
            concat=False,
        )
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, embedding=None):
        out, (new_edge_index, att_weight) = self.gnn(
            x,
            edge_index,
            embedding,
            return_attention_weights=True,
        )
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
        out = self.bn(out)
        return self.relu(out)


class GDNBackbone(nn.Module):
    def __init__(
        self,
        edge_index_sets,
        node_num,
        dim=64,
        out_layer_inter_dim=256,
        input_dim=10,
        out_layer_num=1,
        topk=20,
    ):
        super().__init__()
        self.edge_index_sets = edge_index_sets
        self.embedding = nn.Embedding(node_num, dim)
        self.bn_outlayer_in = nn.BatchNorm1d(dim)
        self.gnn_layers = nn.ModuleList(
            [
                GNNLayer(input_dim, dim, inter_dim=dim + dim, heads=1)
                for _ in range(len(edge_index_sets))
            ]
        )
        self.topk = max(1, min(int(topk), node_num))
        self.learned_graph = None
        self.out_layer = OutLayer(
            dim * len(edge_index_sets),
            out_layer_num,
            inter_num=out_layer_inter_dim,
        )
        self.cache_edge_index_sets = [None] * len(edge_index_sets)
        self.dp = nn.Dropout(0.2)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data):
        x = data
        edge_index_sets = self.edge_index_sets
        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.reshape(-1, all_feature).contiguous()

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_index = edge_index.to(device)
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(
                    edge_index,
                    batch_num,
                    node_num,
                ).to(device)

            all_embeddings = self.embedding(torch.arange(node_num, device=device))
            weights = all_embeddings.detach().clone().view(node_num, -1)
            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(
                weights.norm(dim=-1).view(-1, 1),
                weights.norm(dim=-1).view(1, -1),
            )
            cos_ji_mat = cos_ji_mat / (normed_mat + 1e-8)

            topk_indices_ji = torch.topk(cos_ji_mat, self.topk, dim=-1).indices
            self.learned_graph = topk_indices_ji

            gated_i = (
                torch.arange(0, node_num, device=device)
                .unsqueeze(1)
                .repeat(1, self.topk)
                .flatten()
                .unsqueeze(0)
            )
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
            batch_gated_edge_index = get_batch_edge_index(
                gated_edge_index,
                batch_num,
                node_num,
            ).to(device)
            batch_embeddings = all_embeddings.repeat(batch_num, 1)
            gcn_out = self.gnn_layers[i](
                x,
                batch_gated_edge_index,
                embedding=batch_embeddings,
            )
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        indexes = torch.arange(0, node_num, device=device)
        out = torch.mul(x, self.embedding(indexes))
        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)
        out = self.dp(out)
        out = self.out_layer(out)
        return out.view(-1, node_num)


class GDNModel(nn.Module):
    def __init__(self, win_size: int, n_features: int, hidden_dim: int = 64, topk: int = 8):
        super().__init__()
        edge_index = build_complete_edge_index(n_features)
        self.backbone = GDNBackbone(
            [edge_index],
            node_num=n_features,
            dim=hidden_dim,
            out_layer_inter_dim=max(64, hidden_dim * 4),
            input_dim=win_size,
            out_layer_num=1,
            topk=topk,
        )

    def forward(self, x):
        return self.backbone(x.transpose(1, 2))
