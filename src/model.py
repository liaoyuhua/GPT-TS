"""
Reference: https://github.com/karpathy/minGPT/tree/master.
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from .embedding import TemporalEmbedding


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


def mae_loss(prediction, target):
    mask = ~torch.isnan(target)
    masked_prediction = prediction[mask]
    masked_target = target[mask]
    loss = F.l1_loss(masked_prediction, masked_target)
    return loss


def mse_loss(prediction, target):
    mask = ~torch.isnan(target)
    masked_prediction = prediction[mask]
    masked_target = target[mask]
    loss = F.mse_loss(masked_prediction, masked_target)
    return loss


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(
        self,
        block_size: int,
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        block_size: int,
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            block_size=block_size,
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class GPT2TS(nn.Module):
    """
    TODO: support more model types, e.g. BERT, Gopher, etc.
    """

    models = ["gpt2"]

    params = {
        "gpt2": dict(block_size=1024, n_head=12, n_embd=768),
    }

    def __init__(
        self,
        input_len: int,
        pred_len: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        # patch_size: int,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        tpe_type: str = "fixed",
        freq: str = "h",
        model_type: str = "gpt2",
    ):
        super().__init__()

        self.input_len = input_len
        self.pred_len = pred_len
        assert (
            self.input_len + self.pred_len - 1
        ) <= block_size, f"input_len + pred_len - 1 must be less than or equal to block_size: {block_size}"

        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        # self.patch_size = patch_size
        assert model_type in self.models, f"model_type must be one of {self.models}"
        self.model_type = model_type
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop

        self.tpe = TemporalEmbedding(
            d_model=self.n_embd,
            embed_type=tpe_type,
            freq=freq,
        )
        self.wpe = nn.Embedding(self.input_len + self.pred_len - 1, self.n_embd)
        self.inorm = nn.InstanceNorm1d(1)  # univariate input

        self.transformer = nn.ModuleDict(
            dict(
                # wte=nn.Conv1d(
                #     in_channels=1,
                #     out_channels=self.n_embd,
                #     kernel_size=1,
                #     # stride=self.patch_size,
                # ),  # input embedding
                wte=nn.Linear(1, self.n_embd),
                drop=nn.Dropout(self.embd_pdrop),
                h=nn.ModuleList(
                    [
                        Block(
                            n_embd=self.n_embd,
                            n_head=self.n_head,
                            attn_pdrop=self.attn_pdrop,
                            resid_pdrop=self.resid_pdrop,
                            block_size=self.block_size,
                        )
                        for _ in range(self.n_layer)
                    ]
                ),
                # ln_f=nn.LayerNorm(self.n_embd),
            )
        )

        self.head = nn.Linear(self.n_embd, 1, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, config: dict):
        """
        Initialize a pretrained LLM model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert "model_type" in config, "config must have a `model_type` field"
        if "n_layer" not in config:
            config["n_layer"] = 12  # openai gpt2 default
        model_type = config["model_type"]
        n_layer = config["n_layer"]
        config.update(cls.params[model_type])
        from transformers import GPT2LMHeadModel

        model = GPT2TS(**config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # filer out unnecessary keys and layers deeper than n_layer
        drop_keys = ["wte", "wpe", "lm_head", "ln_f"] + [
            f"h.{i}." for i in range(n_layer, model_hf.config.n_layer)
        ]
        sd_hf = {
            k: v for k, v in sd_hf.items() if not any((k_ in k) for k_ in drop_keys)
        }

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith("attn.masked_bias")]  # ignore these
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        # freeze the multihead attention layers and feedforward
        # layers by default.
        for n, p in model.named_parameters():
            if any(
                k in n for k in ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
            ):
                p.requires_grad = False

        return model

    def forward(self, x, x_mark, y=None):
        """
        Compute the output and loss given the input data.

        Args:
            x: input data, shape (batch_size, input_len+pred_len-1)
            y: target data, shape (batch_size, input_len+pred_len-1), nan for masked
                values which are not used for loss computation
        """
        device = x.device

        b, t, c = x.size()

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)
        pos_emb = self.wpe(pos)
        x_norm = self.inorm(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_norm
        tok_emb = self.transformer.wte(x)
        temp_emb = self.tpe(x_mark)
        h = self.transformer.drop(tok_emb + pos_emb + temp_emb)

        for block in self.transformer.h:
            h = block(h)
        out = self.head(h)

        loss = None, None
        if y is not None:
            loss = self.loss(out, y)

        return out, loss

    @torch.no_grad()
    def predict(self, x, x_mark, pred_mark, y_true=None, pred_len=None):
        """
        Args:
            x: input data, shape (batch_size, input_len, 1)
            x_mark: input mark, shape (batch_size, input_len, n_channel)
            pred_mark: prediction mark, shape (batch_size, pred_len-1, n_channel)
        """
        assert (
            x.dim() == 3 and x_mark.dim() == 3 and pred_mark.dim() == 3
        ), "input shape mismatch"

        pred_len = self.pred_len if pred_len is None else pred_len

        assert pred_mark.shape[1] == (
            pred_len - 1
        ), f"pred_mark shape mismatch: {pred_mark.shape[1]} vs {pred_len-1}"
        for i in range(pred_len):
            out, _ = self(x, x_mark)
            out = out[:, -1, :].unsqueeze(-1)
            x = torch.cat([x, out], dim=1)
            if i < pred_len - 1:
                x_mark = torch.cat([x_mark, pred_mark[:, i : i + 1, :]], dim=1)

        if y_true is not None:
            return x[:, -pred_len:, :], self.loss(x[:, -pred_len:, :], y_true)
        else:
            return x[:, -pred_len:, :]

    @property
    def num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        n_params_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": n_params, "grad": n_params_grad}

    def loss(self, y_pred, y_true):
        return (mse_loss(y_pred, y_true), mae_loss(y_pred, y_true))
