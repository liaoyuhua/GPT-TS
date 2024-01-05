"""
TODO: support LoRA
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from .modules import RevIN, Block, FlattenHead, PoolingHead


class GPT2TS(nn.Module):
    """
    TODO: support more model backbones, e.g. BERT, Gopher, etc.

    Notation:
        B: batch size
        N: number of time series
        E: number of dimensions of embedding
        P: number of patches
        PS: size of the patch
        L: length of input time series
        Y: length of prediction time series
    """

    models = ("gpt2",)
    head_types = ("flatten", "pooling")

    params = {
        "gpt2": dict(block_size=1024, n_head=12, n_embd=768),
    }

    def __init__(
        self,
        num_series: int,
        input_len: int,
        pred_len: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        patch_size: int,
        patch_stride: int,
        revin: bool = True,
        affine: bool = True,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        head_type: str = "flatten",
        individual: bool = False,
        head_pdtop: float = 0.1,
        model_type: str = "gpt2",
    ):
        """
        Args:
            num_series: number of time series, N
            input_len: length of input time series, L
            pred_len: length of prediction time series, Y
            block_size: length of the block, which is the maximum length of the input.
                This is fixed by openai gpt2.
            n_layer: number of transformer layers
            n_head: number of heads in multihead attention
            n_embd: number of dimensions of embedding
            patch_size: size of the patch
            patch_stride: stride of the patch
            revin: whether to use RevIN
            affine: whether to use affine transformation in RevIN
            embd_pdrop: dropout rate for embedding layer
            resid_pdrop: dropout rate for residual connection
            attn_pdrop: dropout rate for attention layer
            head_type: type of the head, must be one of the keys in `head_types`
            head_pdtop: dropout rate for the head
            model_type: type of the model, must be one of the keys in `params`
        """
        super().__init__()
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(num_series, affine=affine)

        self.patch_size = patch_size

        self.patch_stride = patch_stride

        self.patch_num = int((input_len - patch_size) / patch_stride + 1)
        self.patch_padding = False
        # padding to make sure the input length is divisible by patch_stride
        if (input_len - patch_size) % patch_stride != 0:
            self.padding_patch_layer = nn.ReplicationPad1d((0, patch_stride))
            self.patch_num += 1
            self.patch_padding = True

        self.input_len = input_len
        self.pred_len = pred_len
        assert (
            self.patch_num <= block_size
        ), f"patch_num must be less than or equal to block_size: {block_size}"

        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        assert model_type in self.models, f"model_type must be one of {self.models}"
        self.model_type = model_type
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop

        self.wpe = nn.Embedding(self.patch_num, self.n_embd)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(self.patch_size, self.n_embd),  # patch embedding
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

        assert (
            head_type in self.head_types
        ), f"head_type must be one of {self.head_types}"

        if head_type == "flatten":
            self.head = FlattenHead(
                individual=individual,
                n_vars=num_series,
                nf=self.n_embd * self.patch_num,
                target_window=self.pred_len,
                head_dropout=head_pdtop,
            )
        elif head_type == "pooling":
            self.head = PoolingHead(
                n_embd=self.n_embd,
                target_window=self.pred_len,
                head_dropout=head_pdtop,
            )

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
        drop_keys = ["wte", "wpe", "lm_head", "ln_f", "ln_1", "ln_2"] + [
            f"h.{i}." for i in range(n_layer, model_hf.config.n_layer)
        ]
        sd_hf = {
            k: v for k, v in sd_hf.items() if all(k_ not in k for k_ in drop_keys)
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

    def encoder(self, x: torch.Tensor):
        """
        Compute the output of the transformer encoder.
        """
        device = x.device
        B, N, P, PS = x.shape
        # patch embedding
        tok_emb = self.transformer.wte(x)  # B, N, P, E
        tok_emb = torch.reshape(
            tok_emb,
            (tok_emb.shape[0] * tok_emb.shape[1], tok_emb.shape[2], tok_emb.shape[3]),
        )  # B x N, P, E

        # position embedding
        pos = torch.arange(
            0, self.patch_num, dtype=torch.long, device=device
        ).unsqueeze(
            0
        )  # 1, P
        pos_emb = self.wpe(pos)  # 1, P, E

        h = self.transformer.drop(tok_emb + pos_emb)  # B x N, P, E

        for block in self.transformer.h:
            h = block(h)  # B x N, P, E

        h = h.reshape(B, N, P, -1)  # B, N, P, E
        h = h.permute(0, 1, 3, 2)  # B, N, E, P

        return h  # B, N, E, P

    def forward(self, x: torch.Tensor):
        """
        Compute the output and loss given the input data.

        Args:
            x: input data, shape (B, N, L)
        """

        # norm
        if self.revin:
            x = x.permute(0, 2, 1)  # B, L, N
            x = self.revin_layer(x, "norm")  # B, L, N
            x = x.permute(0, 2, 1)  # B, N, L

        # patching
        if self.patch_padding:
            x = self.padding_patch_layer(x)
        x = x.unfold(
            dimension=-1, size=self.patch_size, step=self.patch_stride
        )  # B, N, P, PS

        # encoder
        h = self.encoder(x)  # B, N, E, P
        out = self.head(h)  # B, N, Y

        # denorm
        if self.revin:
            out = out.permute(0, 2, 1)  # B, Y, N
            out = self.revin_layer(out, "denorm")  # B, Y, N
            out = out.permute(0, 2, 1)  # B, N, Y

        return out

    @property
    def num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        n_params_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": n_params, "grad": n_params_grad}
