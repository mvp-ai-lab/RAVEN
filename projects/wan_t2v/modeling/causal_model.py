"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from typing import List, Optional

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from einops import rearrange
from torch.nn.attention.flex_attention import BlockMask

from common.meter import get_running_average_meter
from common.phase import ExecutionPhase, get_execution_phase
from utils.flex_attn import FlexAttention
from utils.naive_cache import NaiveCache

from . import model as wan
from .checkpointing import maybe_checkpoint


@amp.autocast(enabled=False)
def apply_latent_pos_embed(xs, grid_sizes, freqs, frame_shifts=None, packed=False):
    # n, c = x.size(2), x.size(3) // 2
    c = freqs.size(1)
    if frame_shifts is None:
        frame_shifts = [0] * len(grid_sizes)

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    curr = 0
    for i, ((f, h, w), fs) in enumerate(zip(grid_sizes, frame_shifts)):
        seq_len = f * h * w
        if packed:  # xs: [S, D]
            x = xs[curr:curr+seq_len]  # [S, D]
            curr += seq_len
        else:  # xs: list of [1, S, D]
            x = xs[i]  # [1, S, D]

        # precompute multipliers
        x_i = torch.view_as_complex(x.contiguous().to(torch.float64).reshape(
            seq_len, -1, c, 2))  # [seqlen, n_head, head_dim // 2]
        freqs_i = torch.cat([
            freqs[0][fs:fs+f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)  # [seqlen, 1, head_dim // 2]

        # apply rotary embedding
        if packed:
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)  # [seqlen, n_head, head_dim]
        else:
            x_i = torch.view_as_real(x_i * freqs_i).flatten(1)  # [S, D]
        # x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)

    if packed:
        assert curr == xs.size(0), f"Expected curr ({curr}) to equal xs.size(0) ({xs.size(0)})"
        return torch.cat(output).float()
    else:
        return [u.float() for u in output]


class CausalWanSelfAttention(wan.WanSelfAttention):
    def __init__(
        self,
        *args,
        layer_idx,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.layer_idx = layer_idx
        self.flex_attention = FlexAttention()

    def forward(
        self,
        packed_sequence,
        grid_sizes,
        freqs,
        frame_shifts,
        attention_mask,
        q_ranges,
        k_ranges,
        attn_type_map,
        attn_workloads,
        sample_lens,
        past_key_values: Optional[NaiveCache] = None,
        update_past_key_values: bool = False,
        key_value_lens: torch.IntTensor = None,
        packed_query_indexes: Optional[torch.IntTensor] = None,
        packed_past_key_value_indexes: Optional[torch.IntTensor] = None,
    ):
        s, n, d = packed_sequence.size(0), self.num_heads, self.head_dim

        packed_query_states = self.norm_q(self.q(packed_sequence)).view(s, n, d)
        packed_key_states = self.norm_k(self.k(packed_sequence)).view(s, n, d)
        packed_value_states = self.v(packed_sequence).view(s, n, d)

        packed_query_states = apply_latent_pos_embed(packed_query_states, grid_sizes, freqs, frame_shifts, packed=True)
        packed_key_states = apply_latent_pos_embed(packed_key_states, grid_sizes, freqs, frame_shifts, packed=True)

        if past_key_values is not None:  # use flash attn
            if past_key_values.seq_lens(self.layer_idx) > 0:  # merge required
                past_key_states = past_key_values.key_cache[self.layer_idx]
                past_value_states = past_key_values.value_cache[self.layer_idx]
                seqlens = len(packed_query_indexes) + len(packed_past_key_value_indexes)
                merged_key_states = past_key_states.new_zeros(size=[seqlens, n, d])
                merged_value_states = past_value_states.new_zeros(size=[seqlens, n, d])
                merged_key_states[packed_query_indexes] = packed_key_states
                merged_key_states[packed_past_key_value_indexes] = past_key_states
                merged_value_states[packed_query_indexes] = packed_value_states
                merged_value_states[packed_past_key_value_indexes] = past_value_states
                packed_key_states, packed_value_states = merged_key_states, merged_value_states

            if update_past_key_values:
                past_key_values.update_kvcache(self.layer_idx, packed_key_states, packed_value_states, sample_lens[:len(key_value_lens)])

            packed_attn_output = self.flash_attention(
                packed_query_states,
                packed_key_states,
                packed_value_states,
                q_lens=sample_lens[:len(key_value_lens)],
                k_lens=key_value_lens
            )

        else:  # use flex attn
            packed_attn_output = self.flex_attention(
                packed_query_states,
                packed_key_states,
                packed_value_states,
                attention_mask,
                q_ranges,
                k_ranges,
                attn_type_map,
                attn_workloads,
                sample_lens
            )

        return self.o(packed_attn_output.flatten(1))


class CausalWanT2VCrossAttention(wan.WanSelfAttention):
    def __init__(self, *args, layer_idx, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_idx = layer_idx

    def forward(self, x, context, sample_lens,
                past_key_values: Optional[NaiveCache] = None,
                update_past_key_values: bool = False,
                key_value_lens: torch.IntTensor = None,
                packed_new_key_value_indexes: Optional[torch.IntTensor] = None,
                packed_past_key_value_indexes: Optional[torch.IntTensor] = None,
                ):
        b, s, n, d = len(key_value_lens), x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(-1, n, d)

        if context is not None:
            k = self.norm_k(self.k(context)).view(-1, n, d)
            v = self.v(context).view(-1, n, d)

        # kv cache cross attn
        if past_key_values is not None and past_key_values.seq_lens(self.layer_idx) > 0:
            if context is not None:  # merge required
                past_key_states = past_key_values.key_cache[self.layer_idx]
                past_value_states = past_key_values.value_cache[self.layer_idx]
                seqlens = len(packed_new_key_value_indexes) + len(packed_past_key_value_indexes)
                merged_key_states = past_key_states.new_zeros(size=[seqlens, n, d])
                merged_value_states = past_value_states.new_zeros(size=[seqlens, n, d])
                merged_key_states[packed_new_key_value_indexes] = k
                merged_key_states[packed_past_key_value_indexes] = past_key_states
                merged_value_states[packed_new_key_value_indexes] = v
                merged_value_states[packed_past_key_value_indexes] = past_value_states
                k, v = merged_key_states, merged_value_states
            else:
                k = past_key_values.key_cache[self.layer_idx]
                v = past_key_values.value_cache[self.layer_idx]

        if update_past_key_values and context is not None:
            assert all(sink == 0 for sink in past_key_values.sink) and \
                all(window_size is None for window_size in past_key_values.window_size), \
                "Cross-attention cache only supports full cache update."
            past_key_values.key_cache[self.layer_idx] = k
            past_key_values.value_cache[self.layer_idx] = v

        # compute attention
        q_lens, key_value_lens = key_value_lens
        x = self.flash_attention(q, k, v, q_lens=q_lens, k_lens=key_value_lens)

        return self.o(x.flatten(1))


class CausalWanI2VCrossAttention(wan.WanI2VCrossAttention):

    def __init__(self, *args, layer_idx, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_idx = layer_idx

    def forward(self, x, context, sample_lens,
                past_key_values: Optional[List[NaiveCache]] = None,
                update_past_key_values: List[bool] = False,
                key_value_lens: List[torch.IntTensor] = None,
                packed_new_key_value_indexes: Optional[List[torch.IntTensor]] = None,
                packed_past_key_value_indexes: Optional[List[torch.IntTensor]] = None,
                ):
        context, context_img = context
        past_key_values, past_key_values_img = past_key_values
        update_past_key_values, update_past_key_values_img = update_past_key_values
        key_value_lens, key_value_lens_img = key_value_lens
        packed_new_key_value_indexes, packed_new_key_value_indexes_img = packed_new_key_value_indexes
        packed_past_key_value_indexes, packed_past_key_value_indexes_img = packed_past_key_value_indexes
        b, s, n, d = len(key_value_lens), x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(-1, n, d)

        # txt context
        if context is not None:
            k = self.norm_k(self.k(context)).view(-1, n, d)
            v = self.v(context).view(-1, n, d)

        if past_key_values is not None and past_key_values.seq_lens(self.layer_idx) > 0:
            if context is not None:  # merge required
                past_key_states = past_key_values.key_cache[self.layer_idx]
                past_value_states = past_key_values.value_cache[self.layer_idx]
                seqlens = len(packed_new_key_value_indexes) + len(packed_past_key_value_indexes)
                merged_key_states = past_key_states.new_zeros(size=[seqlens, n, d])
                merged_value_states = past_value_states.new_zeros(size=[seqlens, n, d])
                merged_key_states[packed_new_key_value_indexes] = k
                merged_key_states[packed_past_key_value_indexes] = past_key_states
                merged_value_states[packed_new_key_value_indexes] = v
                merged_value_states[packed_past_key_value_indexes] = past_value_states
                k, v = merged_key_states, merged_value_states
            else:
                k = past_key_values.key_cache[self.layer_idx]
                v = past_key_values.value_cache[self.layer_idx]

        if update_past_key_values and context is not None:
            assert all(sink == 0 for sink in past_key_values.sink) and \
                all(window_size is None for window_size in past_key_values.window_size), \
                "Cross-attention cache only supports full cache update."
            past_key_values.key_cache[self.layer_idx] = k
            past_key_values.value_cache[self.layer_idx] = v

        # img context
        if context_img is not None:
            k_img = self.norm_k_img(self.k_img(context_img)).view(-1, n, d)
            v_img = self.v_img(context_img).view(-1, n, d)

        if past_key_values_img is not None and past_key_values_img.seq_lens(self.layer_idx) > 0:
            if context_img is not None:  # merge required
                past_key_states_img = past_key_values_img.key_cache[self.layer_idx]
                past_value_states_img = past_key_values_img.value_cache[self.layer_idx]
                seqlens_img = len(packed_new_key_value_indexes_img) + len(packed_past_key_value_indexes_img)
                merged_key_states_img = past_key_states_img.new_zeros(size=[seqlens_img, n, d])
                merged_value_states_img = past_value_states_img.new_zeros(size=[seqlens_img, n, d])
                merged_key_states_img[packed_new_key_value_indexes_img] = k_img
                merged_key_states_img[packed_past_key_value_indexes_img] = past_key_states_img
                merged_value_states_img[packed_new_key_value_indexes_img] = v_img
                merged_value_states_img[packed_past_key_value_indexes_img] = past_value_states_img
                k_img, v_img = merged_key_states_img, merged_value_states_img
            else:
                k_img = past_key_values_img.key_cache[self.layer_idx]
                v_img = past_key_values_img.value_cache[self.layer_idx]

        if update_past_key_values_img and context_img is not None:
            assert all(sink == 0 for sink in past_key_values_img.sink) and \
                all(window_size is None for window_size in past_key_values_img.window_size), \
                "Cross-attention cache only supports full cache update."
            past_key_values_img.key_cache[self.layer_idx] = k_img
            past_key_values_img.value_cache[self.layer_idx] = v_img

        # compute attention
        img_q_lens, key_value_lens_img = key_value_lens_img
        q_lens, key_value_lens = key_value_lens
        img_x = self.flash_attention(q, k_img, v_img, q_lens=img_q_lens, k_lens=key_value_lens_img)
        x = self.flash_attention(q, k, v, q_lens=q_lens, k_lens=key_value_lens)

        return self.o((x + img_x).flatten(1))


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': CausalWanT2VCrossAttention,
    'i2v_cross_attn': CausalWanI2VCrossAttention,
}


class CausalWanAttentionBlock(nn.Module):
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 layer_idx=None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.layer_idx = layer_idx

        # layers
        self.norm1 = wan.WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, window_size, qk_norm, eps,
                                                layer_idx=layer_idx)
        self.norm3 = wan.WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps, layer_idx=layer_idx)
        self.norm2 = wan.WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        attention_mask,
        q_ranges,
        k_ranges,
        attn_type_map,
        attn_workloads,
        sample_lens,
        # seq_lens,
        grid_sizes,
        freqs,
        frame_shifts,
        context,
        # context_lens,
        # self attn
        past_key_values_self_attn=None,
        update_past_key_values_self_attn=False,
        key_value_lens_self_attn=None,
        packed_query_indexes_self_attn=None,
        packed_past_key_value_indexes_self_attn=None,
        # cross attn
        past_key_values_cross_attn=None,
        update_past_key_values_cross_attn=False,
        key_value_lens_cross_attn=None,
        packed_new_key_value_indexes_cross_attn=None,
        packed_past_key_value_indexes_cross_attn=None,
        # complex indexes
        packed_latent_indexes=None,
    ):
        # The fp32 modulation residuals below promote x to fp32; with gc_step=1
        # every block's checkpoint boundary would then store a full fp32 packed
        # state (14B interleaved: 39 x 1.16 GiB ~= 45.8 GiB, the 8-way smoke
        # OOM). Record the entry dtype and cast back on return so checkpoint
        # boundaries stay bf16 while in-block fp32 numerics are unchanged.
        residual_dtype = x.dtype
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).unbind(dim=1)

        # self-attention
        norm1_x = self.norm1(x).float()
        modulated_norm1_x = norm1_x * (1 + e[1]) + e[0]

        y = self.self_attn(
            modulated_norm1_x, grid_sizes, freqs, frame_shifts,
            attention_mask, q_ranges, k_ranges, attn_type_map, attn_workloads, sample_lens,
            past_key_values_self_attn, update_past_key_values_self_attn, key_value_lens_self_attn,
            packed_query_indexes_self_attn, packed_past_key_value_indexes_self_attn)

        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        _x = x[packed_latent_indexes]
        _x = _x + self.cross_attn(self.norm3(_x), context, sample_lens,
            past_key_values_cross_attn, update_past_key_values_cross_attn, key_value_lens_cross_attn,
            packed_new_key_value_indexes_cross_attn, packed_past_key_value_indexes_cross_attn)
        x = _x

        norm2_x = self.norm2(x).float()
        modulated_norm2_x = norm2_x * (1 + e[4]) + e[3]
        y = self.ffn(modulated_norm2_x)

        with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]

        return x.to(dtype=residual_dtype)


class CausalWanHead(wan.Head):
    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [L, C]
            e(Tensor): Shape [L, C]

            modulation: Shape [1, 2, C]
        """
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).unbind(dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class CausalWanModel(wan.WanModel):
    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 embed_checkpoint_enabled=False,
                 block_checkpoint_enabled=True,
                 block_checkpoint_step=1,
                 block_checkpoint_start_idx=0,
                 guidance_embeds=None,
                 ):
        nn.Module.__init__(self)

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.embed_checkpoint_enabled = embed_checkpoint_enabled
        self.block_checkpoint_enabled = block_checkpoint_enabled
        self.block_checkpoint_step = block_checkpoint_step
        self.block_checkpoint_start_idx = block_checkpoint_start_idx
        self.guidance_embeds = guidance_embeds

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        if self.guidance_embeds:
            self.guidance_embedding = nn.Sequential(
                nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            self.guidance_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        # cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        attn_type = {
            't2v':'t2v_cross_attn',
            'i2v':'i2v_cross_attn',
        }
        cross_attn_type = attn_type[model_type]
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    window_size, qk_norm, cross_attn_norm, eps, layer_idx=i)
            for i in range(num_layers)
        ])

        # head
        self.head = CausalWanHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            wan.rope_params(1024, d - 4 * (d // 6)),
            wan.rope_params(1024, 2 * (d // 6)),
            wan.rope_params(1024, 2 * (d // 6))
        ], dim=1)

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = wan.MLPProj(1280, dim, flf_pos_emb=model_type == 'flf2v')

        # initialize weights
        self.init_weights()

        # update trainable_param_names in the first freeze call
        self.is_first_freeze_call = True
        self.trainable_param_names = []

        self.gradient_checkpointing = False

    def load_state_dict(self, state_dict, *args, **kwargs):
        if "generator" in state_dict:
            state_dict = state_dict["generator"]
        elif "generator_ema" in state_dict:
            state_dict = state_dict["generator_ema"]
        if "model" in state_dict:
            state_dict = state_dict["model"]
        normalized_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("_fsdp_wrapped_module.", "")
            if k.startswith("model."):
                k = k.replace("model.", "", 1)
            if k.startswith("net."):
                k = k.replace("net.", "", 1)
            normalized_state_dict[k] = v
        state_dict = normalized_state_dict
        # Causal-rCM checkpoints store patch_embedding as a flattened linear weight.
        patch_weight = state_dict.get("patch_embedding.weight")
        if (
            patch_weight is not None
            and patch_weight.shape != self.patch_embedding.weight.shape
            and patch_weight.numel() == self.patch_embedding.weight.numel()
        ):
            state_dict["patch_embedding.weight"] = patch_weight.reshape(self.patch_embedding.weight.shape)
        msg = super().load_state_dict(state_dict, *args, **kwargs)
        return msg

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

        # init guidance embeddings
        generator = torch.Generator().manual_seed(42)
        if self.guidance_embeds:
            self.guidance_embedding.to_empty(device="cpu")
            self.guidance_projection.to_empty(device="cpu")
            # nn.init.normal_(self.guidance_embedding[0].weight, std=0.02, generator=generator)
            nn.init.kaiming_normal_(self.guidance_embedding[0].weight, nonlinearity='relu', generator=generator)
            nn.init.zeros_(self.guidance_embedding[0].bias)
            # nn.init.normal_(self.guidance_embedding[2].weight, std=0.02, generator=generator)
            nn.init.kaiming_normal_(self.guidance_embedding[2].weight, nonlinearity='relu', generator=generator)
            nn.init.zeros_(self.guidance_embedding[2].bias)
            nn.init.zeros_(self.guidance_projection[1].weight)
            nn.init.zeros_(self.guidance_projection[1].bias)

    def preprocess_kvcache_cross_attn(
        self,
        context: Optional[torch.Tensor],
        key_value_lens_cross_attn: Optional[torch.IntTensor],
        past_key_values_cross_attn: Optional[NaiveCache],
        update_past_key_values_cross_attn: bool,
    ):
        from .pnp_repeat import pnp_repeat

        if past_key_values_cross_attn is not None:  # inference
            if past_key_values_cross_attn.seq_len > 0:
                past_key_value_len_cross_attn = sum(past_key_values_cross_attn.kvlens)
                past_key_value_lens_cross_attn_tensor = torch.tensor(past_key_values_cross_attn.kvlens, dtype=torch.int32).to(self.device, non_blocking=True)

                if key_value_lens_cross_attn is not None:  # merge required
                    key_value_lens_cumsum = torch.cumsum(past_key_value_lens_cross_attn_tensor, dim=0)
                    key_value_lens_cumsum_repeat = pnp_repeat(key_value_lens_cumsum, repeats=key_value_lens_cross_attn)
                    packed_new_key_value_indexes_cross_attn = torch.arange(context.size(0), device=self.device) + key_value_lens_cumsum_repeat

                    key_value_lens_cross_attn_cumsum = torch.cumsum(key_value_lens_cross_attn, dim=0)
                    key_value_lens_cross_attn_cumsum = torch.cat([torch.tensor([0], device=self.device), key_value_lens_cross_attn_cumsum[:-1]], dim=0)
                    key_value_lens_cross_attn_cumsum_repeat = pnp_repeat(key_value_lens_cross_attn_cumsum, repeats=past_key_value_lens_cross_attn_tensor)
                    packed_past_key_value_indexes_cross_attn = torch.arange(past_key_value_len_cross_attn, device=self.device) + key_value_lens_cross_attn_cumsum_repeat

                    # update merged key_value_lens_cross_attn
                    key_value_lens_cross_attn = key_value_lens_cross_attn + past_key_value_lens_cross_attn_tensor

                else:  # no new context, use only past
                    packed_new_key_value_indexes_cross_attn = None
                    packed_past_key_value_indexes_cross_attn = None
                    key_value_lens_cross_attn = past_key_value_lens_cross_attn_tensor

            else:  # no history, use flash-attn directly
                packed_new_key_value_indexes_cross_attn = None
                packed_past_key_value_indexes_cross_attn = None

            if update_past_key_values_cross_attn:
                # assert past_key_values_cross_attn.sink == 0 and past_key_values_cross_attn.window_size is None, \
                assert all(sink == 0 for sink in past_key_values_cross_attn.sink) and \
                    all(window_size is None for window_size in past_key_values_cross_attn.window_size), \
                    f"Only non-windowed cross-attention with sink=0 is supported for kv cache update."
                past_key_values_cross_attn.kvlens = key_value_lens_cross_attn.tolist()

        else:  # training
            packed_new_key_value_indexes_cross_attn = None
            packed_past_key_value_indexes_cross_attn = None

        return key_value_lens_cross_attn, packed_new_key_value_indexes_cross_attn, packed_past_key_value_indexes_cross_attn

    def forward(
        self,
        x,
        t,
        context,
        # seq_len,
        packed_position_ids: torch.IntTensor,
        packed_latent_indexes: torch.IntTensor,
        packed_latent_seqlens: torch.IntTensor,
        packed_noisy_latent_relative_indexes: torch.IntTensor,
        packed_noisy_latent_seqlens: torch.IntTensor,
        sample_lens: List[int],
        frame_shifts: List[int],
        attention_mask: Optional[BlockMask] = None,
        q_ranges: Optional[torch.IntTensor] = None,
        k_ranges: Optional[torch.IntTensor] = None,
        attn_type_map: Optional[torch.IntTensor] = None,
        attn_workloads: Optional[List[int]] = None,
        past_key_values_self_attn: Optional[NaiveCache] = None,
        update_past_key_values_self_attn: bool = False,
        past_key_values_cross_attn: Optional[NaiveCache] = None,
        update_past_key_values_cross_attn: bool = False,
        past_key_values_cross_attn_img: Optional[NaiveCache] = None,
        update_past_key_values_cross_attn_img: bool = False,
        clip_fea=None,
        y=None,
        guidance=None,
    ):
        from .pnp_repeat import pnp_repeat

        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.int32) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]  # [1, C, T, H, W] -> [1, S, D]
        # seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        # assert seq_lens.max() <= seq_len
        # x = torch.cat([
        #     torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
        #               dim=1) for u in x
        # ])

        packed_latent_tokens = torch.cat(x, dim=1)[0]  # [S, D]

        packed_sequence = packed_latent_tokens.new_zeros(size=(len(packed_position_ids), self.dim))
        packed_sequence[packed_latent_indexes] = packed_latent_tokens

        sample_lens = torch.tensor(sample_lens, dtype=torch.int32, device=device)
        bsz = len(context) if context is not None and len(context) > 0 else len(x)

        # preprocess kv cache self attn
        if past_key_values_self_attn is not None:  # inference
            if past_key_values_self_attn.seq_len > 0:  # merge required
                query_len, device = packed_sequence.shape[0], packed_sequence.device
                query_lens_tensor = sample_lens[:bsz]
                past_key_value_len = sum(past_key_values_self_attn.kvlens)
                past_key_value_lens_tensor = torch.tensor(past_key_values_self_attn.kvlens, dtype=torch.int32, device=device)

                key_value_lens_cumsum = torch.cumsum(past_key_value_lens_tensor, dim=0)
                key_value_lens_cumsum_repeat = pnp_repeat(key_value_lens_cumsum, repeats=query_lens_tensor)
                packed_query_indexes_self_attn = torch.arange(query_len, device=device) + key_value_lens_cumsum_repeat

                query_lens_cumsum = torch.cumsum(query_lens_tensor, dim=0)
                query_lens_cumsum = torch.cat([torch.tensor([0], device=device), query_lens_cumsum[:-1]], dim=0)
                query_lens_cumsum_repeat = pnp_repeat(query_lens_cumsum, repeats=past_key_value_lens_tensor)
                packed_past_key_value_indexes_self_attn = torch.arange(past_key_value_len, device=device) + query_lens_cumsum_repeat

                # update merged key_value_lens_self_attn
                key_value_lens_self_attn = torch.cat([
                    sample_lens[i:i+1] + past_key_values_self_attn.kvlens[i] for i in range(len(sample_lens))
                ])

                past_position_ids = past_key_values_self_attn.curr_rope  # [bsz,], i.e. curr_rope
                past_position_ids = pnp_repeat(past_position_ids, repeats=query_lens_tensor)
                packed_position_ids = packed_position_ids + past_position_ids

            else:  # use flash-attn but w/o history
                packed_query_indexes_self_attn = None
                packed_past_key_value_indexes_self_attn = None
                key_value_lens_self_attn = sample_lens[:bsz]

            if update_past_key_values_self_attn:  # in-place update
                # past_key_values_self_attn.kvlens = key_value_lens_self_attn.tolist()
                past_key_values_self_attn.update_kvlens(sample_lens[:bsz].tolist())
                position_ids = packed_position_ids.split(sample_lens[:bsz].tolist())
                curr_rope = [position_ids[i][-1] + 1 for i in range(bsz)]
                past_key_values_self_attn.curr_rope = torch.tensor(curr_rope).to(device, non_blocking=True)
        else:  # training
            packed_query_indexes_self_attn = None
            packed_past_key_value_indexes_self_attn = None
            key_value_lens_self_attn = None

        def time_emb(t, packed_latent_indexes, packed_noisy_latent_relative_indexes, packed_noisy_latent_seqlens):
            packed_timesteps = t.new_zeros(size=(len(packed_latent_indexes),))
            if len(packed_noisy_latent_relative_indexes) > 0:
                packed_timesteps[packed_noisy_latent_relative_indexes] = pnp_repeat(
                    t, repeats=packed_noisy_latent_seqlens)
            with amp.autocast(dtype=torch.float32):
                e = self.time_embedding(
                    wan.sinusoidal_embedding_1d(self.freq_dim, packed_timesteps).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            return e, e0

        e, e0 = maybe_checkpoint(
            time_emb,
            t, packed_latent_indexes, packed_noisy_latent_relative_indexes, packed_noisy_latent_seqlens,
            enabled=self.gradient_checkpointing and self.embed_checkpoint_enabled,
        )
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32
        if get_execution_phase() is ExecutionPhase.TRAIN_FORWARD:
            get_running_average_meter().put_scalar("running/time_emb/std", e0.std().item())

        # guidance embeddings
        if self.guidance_embeds:
            def guidance_emb(e0, guidance, packed_latent_indexes, packed_noisy_latent_relative_indexes, packed_noisy_latent_seqlens):
                packed_guidance = t.new_zeros(
                    size=(len(packed_latent_indexes),),
                    dtype=guidance.dtype if guidance is not None else torch.float32,
                )
                if len(packed_noisy_latent_relative_indexes) > 0:
                    assert guidance is not None, f"guidance embeddings required for guidance_embeds=True"
                    guidance = guidance * 1000
                    packed_guidance[packed_noisy_latent_relative_indexes] = pnp_repeat(
                        guidance, repeats=packed_noisy_latent_seqlens)
                with amp.autocast(dtype=torch.float32):
                    g = self.guidance_embedding(
                        wan.sinusoidal_embedding_1d(self.freq_dim, packed_guidance).float())
                    g0 = self.guidance_projection(g).unflatten(1, (6, self.dim))
                    if get_execution_phase() is ExecutionPhase.TRAIN_FORWARD:
                        get_running_average_meter().put_scalar("running/guidance_embeds/std", g0.std().item())
                    e0 = e0 + g0
                return e0

            e0 = maybe_checkpoint(
                guidance_emb,
                e0, guidance, packed_latent_indexes, packed_noisy_latent_relative_indexes, packed_noisy_latent_seqlens,
                enabled=self.gradient_checkpointing and self.embed_checkpoint_enabled,
            )

        # context
        if context is not None and len(context) > 0:
            key_value_lens_cross_attn = torch.tensor(
                [self.text_len for _ in context], dtype=torch.int32, device=device)
            context = self.text_embedding(
                torch.cat([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))
        else:
            context = None
            key_value_lens_cross_attn = None

        key_value_lens_cross_attn, packed_new_key_value_indexes_cross_attn, packed_past_key_value_indexes_cross_attn = self.preprocess_kvcache_cross_attn(
            context, key_value_lens_cross_attn, past_key_values_cross_attn, update_past_key_values_cross_attn)
        q_lens = []  # only latent tokens should attend to text/img/audio in cross-attn
        sample_lens_cumsum = torch.cat([torch.tensor([0], device=device), torch.cumsum(sample_lens, dim=0)], dim=0)
        for i in range(bsz):
            sample_idxs = torch.arange(sample_lens_cumsum[i], sample_lens_cumsum[i+1], device=device)
            sample_latent_indexes = packed_latent_indexes[torch.isin(packed_latent_indexes, sample_idxs)]
            q_lens.append(len(sample_latent_indexes))
        q_lens = torch.tensor(q_lens, dtype=torch.int32, device=device)
        key_value_lens_cross_attn = (q_lens, key_value_lens_cross_attn)

        if self.model_type=="i2v":  # img context for i2v
            if clip_fea is not None and len(clip_fea) > 0:
                context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
                key_value_lens_cross_attn_img = torch.tensor(
                    [context_clip.size(1) for _ in context_clip], dtype=torch.int32, device=device)
                context_clip = context_clip.view(-1, context_clip.size(2))
            else:
                context_clip = None
                key_value_lens_cross_attn_img = None

            key_value_lens_cross_attn_img, packed_new_key_value_indexes_cross_attn_img, packed_past_key_value_indexes_cross_attn_img = self.preprocess_kvcache_cross_attn(
                context_clip, key_value_lens_cross_attn_img, past_key_values_cross_attn_img, update_past_key_values_cross_attn_img)
            key_value_lens_cross_attn_img = (q_lens, key_value_lens_cross_attn_img)

            context = [context, context_clip]
            key_value_lens_cross_attn = [key_value_lens_cross_attn, key_value_lens_cross_attn_img]
            packed_new_key_value_indexes_cross_attn = [packed_new_key_value_indexes_cross_attn, packed_new_key_value_indexes_cross_attn_img]
            packed_past_key_value_indexes_cross_attn = [packed_past_key_value_indexes_cross_attn, packed_past_key_value_indexes_cross_attn_img]
            past_key_values_cross_attn = [past_key_values_cross_attn, past_key_values_cross_attn_img]
            update_past_key_values_cross_attn = [update_past_key_values_cross_attn, update_past_key_values_cross_attn_img]

        # arguments
        kwargs = dict(
            e=e0,
            attention_mask=attention_mask,  # flex attn
            q_ranges=q_ranges,              # magi attn
            k_ranges=k_ranges,              # magi attn
            attn_type_map=attn_type_map,    # magi attn
            attn_workloads=attn_workloads,
            sample_lens=sample_lens,
            # seq_lens=seq_lens,
            grid_sizes=grid_sizes.tolist(),
            freqs=self.freqs,
            frame_shifts=frame_shifts,
            context=context,
            # context_lens=context_lens,
            # self attn
            past_key_values_self_attn=past_key_values_self_attn,
            update_past_key_values_self_attn=update_past_key_values_self_attn,
            key_value_lens_self_attn=key_value_lens_self_attn,
            packed_query_indexes_self_attn=packed_query_indexes_self_attn,
            packed_past_key_value_indexes_self_attn=packed_past_key_value_indexes_self_attn,
            # cross attn
            past_key_values_cross_attn=past_key_values_cross_attn,
            update_past_key_values_cross_attn=update_past_key_values_cross_attn,
            key_value_lens_cross_attn=key_value_lens_cross_attn,
            packed_new_key_value_indexes_cross_attn=packed_new_key_value_indexes_cross_attn,
            packed_past_key_value_indexes_cross_attn=packed_past_key_value_indexes_cross_attn,
            # complex indexes
            packed_latent_indexes=packed_latent_indexes,
        )

        packed_sequence = maybe_checkpoint(
            self.blocks,
            packed_sequence,
            enabled=self.gradient_checkpointing and self.block_checkpoint_enabled,
            gc_step=self.block_checkpoint_step,
            gc_start_idx=self.block_checkpoint_start_idx,
            **kwargs
        )

        # head
        x = self.head(packed_sequence[packed_latent_indexes], e)

        # unpatchify
        xs = x.split(packed_latent_seqlens.tolist(), dim=0)
        x = self.unpatchify(xs, grid_sizes)
        return [u.float() for u in x]
