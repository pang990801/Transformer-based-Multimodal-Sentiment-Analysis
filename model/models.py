import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class Identity(nn.Module):
    """Identity Module."""

    def __init__(self):

        super().__init__()

    def forward(self, x):

        return x


class Transformer(nn.Module):


    def __init__(self, n_features, dim):

        super().__init__()
        self.embed_dim = dim
        self.conv = nn.Conv1d(n_features, self.embed_dim,
                              kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)

    def forward(self, x):

        if type(x) is list:
            x = x[0]
        x = self.conv(x.permute([0, 2, 1]))
        x = x.permute([2, 0, 1])
        x = self.transformer(x)[-1]
        return x


class TransformerSeq(nn.Module):


    def __init__(self, n_features, dim):

        super().__init__()
        self.embed_dim = dim
        self.conv = nn.Conv1d(n_features, self.embed_dim,
                              kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)

    def forward(self, x):

        if type(x) is list:
            x = x[0]
        # Apply 1D convolution to adjust feature dimension
        x = self.conv(x.permute([0, 2, 1]))  # (batch_size, embed_dim, seq_len)
        x = x.permute([2, 0, 1])  # (seq_len, batch_size, embed_dim)
        # Apply transformer encoder
        x = self.transformer(x)  # (seq_len, batch_size, embed_dim)
        x = x.permute([1, 0, 2])  # (batch_size, seq_len, embed_dim)
        return x

class LSTM(torch.nn.Module):


    def __init__(self, indim, hiddim, linear_layer_outdim=None, dropout=False, dropoutp=0.1, flatten=False,
                 has_padding=False):

        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(indim, hiddim, batch_first=True)
        if linear_layer_outdim is not None:
            self.linear = nn.Linear(hiddim, linear_layer_outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.flatten = flatten
        self.has_padding = has_padding
        self.linear_layer_outdim = linear_layer_outdim

    def forward(self, x):

        if self.has_padding:
            x = pack_padded_sequence(
                x[0], x[1], batch_first=True, enforce_sorted=False)
            out = self.lstm(x)[1][0]
        else:
            if len(x.size()) == 2:
                x = x.unsqueeze(2)
            out = self.lstm(x)[1][0]
        out = out.permute([1, 2, 0])
        out = out.reshape([out.size()[0], -1])
        if self.dropout:
            out = self.dropout_layer(out)
        if self.flatten:
            out = torch.flatten(out, 1)
        if self.linear_layer_outdim is not None:
            out = self.linear(out)
        return out


class MLP(torch.nn.Module):


    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1, output_each_layer=False):

        super(MLP, self).__init__()
        self.fc = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):

        output = F.relu(self.fc(x))
        if self.dropout:
            output = self.dropout_layer(output)
        output2 = self.fc2(output)
        if self.dropout:
            output2 = self.dropout_layer(output)
        if self.output_each_layer:
            return [0, x, output, self.lklu(output2)]
        return output2


class GRU(torch.nn.Module):


    def __init__(self, indim, hiddim, dropout=False, dropoutp=0.1, flatten=False, has_padding=False, last_only=False,
                 batch_first=True):

        super(GRU, self).__init__()
        self.gru = nn.GRU(indim, hiddim, batch_first=True)
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.flatten = flatten
        self.has_padding = has_padding
        self.last_only = last_only
        self.batch_first = batch_first

    def forward(self, x):

        if self.has_padding:
            x = pack_padded_sequence(
                x[0], x[1], batch_first=self.batch_first, enforce_sorted=False)
            out = self.gru(x)[1][-1]
        elif self.last_only:
            out = self.gru(x)[1][0]

            return out
        else:
            out, l = self.gru(x)
        if self.dropout:
            out = self.dropout_layer(out)
        if self.flatten:
            out = torch.flatten(out, 1)

        return out


class GRUWithLinear(torch.nn.Module):


    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1, flatten=False, has_padding=False,
                 output_each_layer=False, batch_first=False):

        super(GRUWithLinear, self).__init__()
        self.gru = nn.GRU(indim, hiddim, batch_first=batch_first)
        self.linear = nn.Linear(hiddim, outdim)
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.flatten = flatten
        self.has_padding = has_padding
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):

        if self.has_padding:
            x = pack_padded_sequence(
                x[0], x[1], batch_first=True, enforce_sorted=False)
            hidden = self.gru(x)[1][-1]
        else:
            hidden = self.gru(x)[0]
        if self.dropout:
            hidden = self.dropout_layer(hidden)
        out = self.linear(hidden)
        if self.flatten:
            out = torch.flatten(out, 1)
        if self.output_each_layer:
            return [0, torch.flatten(x, 1), torch.flatten(hidden, 1), self.lklu(out)]
        return out


class Linear(torch.nn.Module):


    def __init__(self, indim, outdim, xavier_init=False):

        super(Linear, self).__init__()
        self.fc = nn.Linear(indim, outdim)
        if xavier_init:
            nn.init.xavier_normal(self.fc.weight)
            self.fc.bias.data.fill_(0.0)

    def forward(self, x):

        return self.fc(x)


class TransformerFusion(nn.Module):
    """Fusion of multiple modalities using a Transformer."""

    def __init__(self, d_model, nhead, num_layers=2, dropout=0.1):
        """
        Args:
            d_model (int): The dimension of the input features.
            nhead (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate.
        """
        super(TransformerFusion, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, d_model)  # Optional output projection layer

    def forward(self, modalities):
        """
        Forward Pass of Transformer Fusion.

        Args:
            modalities (List[torch.Tensor]): List of modality features [batch_size, feature_dim].

        Returns:
            torch.Tensor: Fused feature representation.
        """
        # Stack and permute modalities to match Transformer input format
        modalities = torch.stack(modalities, dim=0)  # [num_modalities, batch_size, feature_dim]
        modalities = modalities.permute(1, 0, 2)  # [batch_size, num_modalities, feature_dim]

        fused = self.transformer(modalities)  # [batch_size, num_modalities, feature_dim]
        fused = torch.mean(fused, dim=1)  # Pool across modalities (mean pooling)

        return self.fc_out(fused)  # Optional projection


class ConcatLate(nn.Module):

    def __init__(self):
        super(ConcatLate, self).__init__()

    def forward(self, modalities):
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))

        return torch.cat(flattened, dim=1)


class ConcatEarly(nn.Module):
    def __init__(self):
        super(ConcatEarly, self).__init__()
    def forward(self, modalities):
        return torch.cat(modalities, dim=2)


class TensorFusion(nn.Module):
    """
    Implementation of TensorFusion Networks.

    See https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py for more and the original code.
    """

    def __init__(self):
        """Instantiates TensorFusion Network Module."""
        super().__init__()

    def forward(self, modalities):
        """
        Forward Pass of TensorFusion.

        :param modalities: An iterable of modalities to combine.
        """
        if len(modalities) == 1:
            return modalities[0]

        mod0 = modalities[0]
        nonfeature_size = mod0.shape[:-1]

        m = torch.cat((Variable(torch.ones(
            *nonfeature_size, 1).type(mod0.dtype).to(mod0.device), requires_grad=False), mod0), dim=-1)
        for mod in modalities[1:]:
            mod = torch.cat((Variable(torch.ones(
                *nonfeature_size, 1).type(mod.dtype).to(mod.device), requires_grad=False), mod), dim=-1)
            fused = torch.einsum('...i,...j->...ij', m, mod)
            m = fused.reshape([*nonfeature_size, -1])

        return m


class LowRankTensorFusion(nn.Module):
    """
    Implements Low-Rank Multimodal Fusion with support for an arbitrary number of modalities.
    Reference: https://github.com/Justin1904/Low-rank-Multimodal-Fusion
    """

    def __init__(self, input_dims, output_dim, rank, flatten=True):
        super(LowRankTensorFusion, self).__init__()

        self.input_dims = input_dims
        self.output_dim = output_dim
        self.rank = rank
        self.flatten = flatten

        # Initialize low-rank factors for each input dimension
        self.factors = nn.ParameterList([
            nn.Parameter(torch.randn(rank, input_dim + 1, output_dim))
            for input_dim in input_dims
        ])

        # Fusion weights and bias
        self.fusion_weights = nn.Parameter(torch.randn(1, rank))
        self.fusion_bias = nn.Parameter(torch.zeros(1, output_dim))

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        for factor in self.factors:
            nn.init.xavier_normal_(factor)
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, modalities):

        batch_size = modalities[0].shape[0]
        fused_tensor = 1
        for (modality, factor) in zip(modalities, self.factors):
            ones = Variable(torch.ones(batch_size, 1).type(
                modality.dtype), requires_grad=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            if self.flatten:
                modality_withones = torch.cat(
                    (ones, torch.flatten(modality, start_dim=1)), dim=1)
            else:
                modality_withones = torch.cat((ones, modality), dim=1)
            modality_factor = torch.matmul(modality_withones, factor)
            fused_tensor = fused_tensor * modality_factor

        output = torch.matmul(self.fusion_weights, fused_tensor.permute(
            1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output

def Linear(in_features, out_features, bias=True):

    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class EarlyFusionTransformer(nn.Module):


    embed_dim = 32

    def __init__(self, n_features):

        super().__init__()
        self.conv = nn.Conv1d(n_features, self.embed_dim,
                              kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.linear = nn.Linear(self.embed_dim, 1)


    def forward(self, x):

        # Conv1D expects (batch_size, n_features, seq_len)
        if isinstance(x, list):
            x = torch.cat(x, dim=2)  # 拼接: (batch_size, seq_len, total_feature_dim)
        x = self.conv(x.permute([0, 2, 1]))  # (batch_size, embed_dim, seq_len)
        x = x.permute([2, 0, 1])  # (seq_len, batch_size, embed_dim)
        x = self.transformer(x)[-1]  # Transformer 输出最后一个时间步的特征

        return x



class LateFusionTransformer(nn.Module):


    def __init__(self, in_dim=1216, embed_dim=32):

        super().__init__()
        self.embed_dim = embed_dim

        # Conv1D 适应输入特征数（多模态特征拼接后）
        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=self.embed_dim,
            kernel_size=1,
            padding=0,
            bias=False
        )
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)


    def forward(self, x):

        # 检查输入是否是列表（模态分开）

        if isinstance(x, list):
            x = torch.cat(x, dim=-1)  # 按特征维度拼接: [batch_size, seq_len, total_feature_dim]

        # Conv1D expects (batch_size, feature_dim, seq_len)
        x = self.conv(x.permute(0, 2, 1))  # 转换为 [batch_size, embed_dim, seq_len]
        x = x.permute(2, 0, 1)  # 转换为 Transformer 输入: [seq_len, batch_size, embed_dim]
        x = self.transformer(x)  # Transformer 输出: [seq_len, batch_size, embed_dim]

        # 返回最后一个时间步的输出
        return x[-1]


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.positional_embedding = SinusoidalPositionalEmbedding(embed_dim)
        self.dropout = embed_dropout
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, attn_dropout, relu_dropout, res_dropout, attn_mask)
            for _ in range(layers)
        ])
        self.layer_norm = LayerNorm(embed_dim) if attn_mask else None

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        x = x_in * self.embed_scale + self.positional_embedding(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            x_k = x_in_k * self.embed_scale + self.positional_embedding(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            x_v = x_in_v * self.embed_scale + self.positional_embedding(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        for layer in self.layers:
            x = layer(x, x_k, x_v) if x_in_k is not None and x_in_v is not None else layer(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        return x
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_mask = attn_mask

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout)
        self.fc1 = Linear(embed_dim, 4 * embed_dim)
        self.fc2 = Linear(4 * embed_dim, embed_dim)

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.norms = nn.ModuleList([LayerNorm(embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        if self.normalize_before:
            x = self.norms[0](x)
            if x_k is not None:
                x_k = self.norms[0](x_k)
            if x_v is not None:
                x_v = self.norms[0](x_v)

        # Apply attention
        x = self.apply_attention(x, x_k, x_v)

        # Apply first sublayer (self-attention + residual)
        x = self.apply_sublayer(x, self.norms[0])

        # Apply second sublayer (feed-forward network + residual)
        x = self.apply_feed_forward(x, self.norms[1])

        return x

    def apply_attention(self, x, x_k, x_v):
        key = x_k if x_k is not None else x
        value = x_v if x_v is not None else x
        attn_mask = self.create_attention_mask(x, key) if self.attn_mask else None
        x, _ = self.self_attn(x, key, value, attn_mask=attn_mask)
        return F.dropout(x, p=self.res_dropout, training=self.training)

    def apply_sublayer(self, x, norm):
        """Applies residual connection followed by layer normalization."""
        return norm(x + F.dropout(x, p=self.res_dropout, training=self.training))

    def apply_feed_forward(self, x, norm):
        """Applies a feed-forward layer with ReLU activation and residual connection."""
        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        return norm(residual + F.dropout(x, p=self.res_dropout, training=self.training))

    def create_attention_mask(self, x, x_k):
        """Create an attention mask, if necessary."""
        # Implementation of mask creation would go here
        return None


def fill_with_neg_inf(tensor):
    """ Fills the input tensor with negative infinity values. """
    return tensor.float().fill_(float('-inf')).type_as(tensor)


def buffered_future_mask(size1, size2=None, device=None):
    """ Generates a future mask for attention mechanisms. """
    if size2 is None:
        size2 = size1
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(size1, size2)), diagonal=1)
    if device:
        future_mask = future_mask.to(device)
    return future_mask


class CustomLinear(nn.Module):
    """ Custom linear layer with Xavier uniform initialization. """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.)

    def forward(self, x):
        return self.linear(x)


class CustomLayerNorm(nn.Module):
    """ Custom layer normalization with potential for future customization. """

    def __init__(self, normalized_shape):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        return self.norm(x)


def make_positions(tensor, padding_idx, left_pad=False):
    """ Generates positions based on the presence of padding in the tensor. """
    max_pos = padding_idx + 1 + tensor.size(1)
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = tensor.new_empty(0)
    if make_positions.range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
    mask = tensor.ne(padding_idx)
    positions = make_positions.range_buf[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - tensor.size(1) + mask.sum(dim=1, dtype=torch.long).unsqueeze(1)
    return tensor.masked_scatter(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """ Sinusoidal positional embedding module. """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.register_buffer('weights', None)

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """ Creates sinusoidal embeddings. """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """ Applies positional embeddings to the input tensor. """
        bsz, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            ).to(input.device)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

class AttentionPooling(nn.Module):
    """注意力加权时序聚合"""
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: [B, T, F]
        attn_weights = torch.softmax(self.attention(x).squeeze(-1), dim=1)  # [B, T]
        out = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [B, F]
        return out


class GatedMultiTransfomerModel(nn.Module):
    class DefaultHyperParams():
        num_heads = 3
        layers = 3
        attn_dropout = 0.1
        attn_dropout_modalities = [0.0] * 1000
        relu_dropout = 0.1
        res_dropout = 0.1
        out_dropout = 0.0
        embed_dropout = 0.25
        embed_dim = 9
        attn_mask = True
        output_dim = 1
        all_steps = False


    def __init__(self, n_modalities, n_features, hyp_params=DefaultHyperParams):
        super().__init__()
        self.n_modalities = n_modalities
        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_modalities = hyp_params.attn_dropout_modalities
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        self.all_steps = hyp_params.all_steps
        self.alpha = nn.Parameter(torch.tensor(0.2))  # 可学习参数，初始值为 0.5
        combined_dim = self.embed_dim * n_modalities
        output_dim = hyp_params.output_dim

        # 模态输入特征映射到 embed_dim
        self.modal_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_features[i], self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(p=self.embed_dropout),
                nn.ReLU(inplace=True)
            )
            for i in range(n_modalities)
        ])

        # Crossmodal Attention 和 Self Attention
        self.trans = nn.ModuleList([
            nn.ModuleList([
                self.get_network(i, j, mem=False) for j in range(n_modalities)
            ]) for i in range(n_modalities)
        ])
        self.trans_mems = nn.ModuleList([
            self.get_network(i, i, mem=True) for i in range(n_modalities)
        ])

        # 模态加权融合参数
        self.modal_weights = nn.Parameter(torch.ones(self.n_modalities))

        # 门控机制
        self.gating_linears = nn.ModuleList([
            nn.Linear(self.embed_dim, self.embed_dim) for _ in range(n_modalities)
        ])

        # Attention Pooling 用于时序聚合
        self.attn_pooling = AttentionPooling(self.embed_dim * self.n_modalities)

        # 分类头
        self.classification_head = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.out_dropout),
            nn.Linear(combined_dim // 2, output_dim)
        )

    def get_network(self, mod1, mod2, mem):
        embed_dim = self.embed_dim
        attn_dropout = self.attn_dropout_modalities[mod2] if not mem else self.attn_dropout
        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=self.layers,
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask
        )

    def forward(self, x):
        proj_x = []
        for i in range(self.n_modalities):
            xi = x[i].permute(1, 0, 2)  # [T, B, F]
            T, B, F = xi.size()
            xi = xi.reshape(T * B, F)
            xi = self.modal_proj[i](xi)
            xi = xi.reshape(T, B, self.embed_dim)
            proj_x.append(xi)

        hs = []
        for i in range(self.n_modalities):
            h_list = []
            for j in range(self.n_modalities):
                h_ij = self.trans[i][j](proj_x[i], proj_x[j], proj_x[j])
                h_list.append(h_ij)
            #h_list = self.trans_mems[i](h_list)

            # 模态加权融合
            modal_weights = torch.softmax(self.modal_weights, dim=0)
            h_fused = torch.stack(h_list, dim=0)
            h_fused = torch.einsum('m,m...->...', modal_weights, h_fused)

            # 门控机制
            gate = torch.sigmoid(self.gating_linears[i](h_fused))
            h_fused = gate * h_fused

            hs.append(h_fused.permute(1, 0, 2))  # [B, T, embed_dim]

        out = torch.cat(hs, dim=2)  # [B, T, combined_dim]

        # Attention Pooling 进行时序聚合
        out = self.attn_pooling(out)  # [B, combined_dim]
        # 分类头
        out_residual = out
        out = self.classification_head(out)
        #out_residual = nn.Linear(192, 1).to(out.device)(out_residual)  # 线性映射

        #self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始值为 0.5
        # 残差连接
        #out = self.alpha * out + (1 - self.alpha) * out_residual  # [B, 1]
        return out

