"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import logging

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config

from . import model as wan
from .causal_model import apply_latent_pos_embed
from .checkpointing import maybe_checkpoint

logger = logging.getLogger()


class PackedWanSelfAttention(wan.WanSelfAttention):
    def forward(self, x, seq_lens, grid_sizes, freqs):
        s, n, d = x.size(0), self.num_heads, self.head_dim

        # query, key, value function
        packed_query_states = self.norm_q(self.q(x)).view(s, n, d)
        packed_key_states = self.norm_k(self.k(x)).view(s, n, d)
        packed_value_states = self.v(x).view(s, n, d)

        packed_query_states = apply_latent_pos_embed(packed_query_states, grid_sizes, freqs, packed=True)
        packed_key_states = apply_latent_pos_embed(packed_key_states, grid_sizes, freqs, packed=True)

        x = self.flash_attention(
            q=packed_query_states,
            k=packed_key_states,
            v=packed_value_states,
            q_lens=seq_lens,
            k_lens=seq_lens
        )

        # output
        x = x.flatten(1)
        x = self.o(x)
        return x


class PackedWanT2VCrossAttention(wan.WanT2VCrossAttention):
    def forward(self, x, seq_lens, context, context_lens):
        b, s, n, d = len(context_lens), x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(-1, n, d)
        k = self.norm_k(self.k(context)).view(-1, n, d)
        v = self.v(context).view(-1, n, d)

        # compute attention
        x = self.flash_attention(q, k, v, q_lens=seq_lens, k_lens=context_lens)

        # output
        x = x.flatten(1)
        x = self.o(x)
        return x


class PackedWanI2VCrossAttention(wan.WanI2VCrossAttention):
    def forward(self, x, seq_lens, context, context_lens):
        context, context_img = context
        context_lens, context_lens_img = context_lens
        b, s, n, d = len(context_lens), x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(-1, n, d)
        k = self.norm_k(self.k(context)).view(-1, n, d)
        v = self.v(context).view(-1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(-1, n, d)
        v_img = self.v_img(context_img).view(-1, n, d)

        # compute attention
        img_x = self.flash_attention(q, k_img, v_img, q_lens=seq_lens, k_lens=context_lens_img)
        x = self.flash_attention(q, k, v, q_lens=seq_lens, k_lens=context_lens)

        # output
        x = x.flatten(1)
        img_x = img_x.flatten(1)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': PackedWanT2VCrossAttention,
    'i2v_cross_attn': PackedWanI2VCrossAttention,
}


class PackedWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = wan.WanLayerNorm(dim, eps)
        self.self_attn = PackedWanSelfAttention(
            dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = wan.WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
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
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).unbind(dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        x = x + self.cross_attn(self.norm3(x), seq_lens, context, context_lens)
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]

        return x


class PackedWanHead(wan.Head):
    def forward(self, x, e):
        r"""
            x(Tensor): Shape [L, C]
            e(Tensor): Shape [L, C]

            modulation: Shape [1, 2, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).unbind(dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class PackedWanModel(wan.WanModel):
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
                 block_checkpoint_enabled=True,
                 block_checkpoint_step=1,
                 block_checkpoint_start_idx=0,
                 guidance_embeds=False
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
            PackedWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = PackedWanHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            wan.rope_params(1024, d - 4 * (d // 6)),
            wan.rope_params(1024, 2 * (d // 6)),
            wan.rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = wan.MLPProj(1280, dim, flf_pos_emb=model_type == 'flf2v')

        # initialize weights
        self.init_weights()

        # update trainable_param_names in the first freeze call
        self.is_first_freeze_call = True
        self.trainable_param_names = []

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        guidance=None
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        from .pnp_repeat import pnp_repeat

        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            assert clip_fea is not None and y is not None

        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.int32, device=device)
        x = torch.cat(x, dim=1)[0]  # [S, D]
        # assert seq_lens.max() <= seq_len
        # x = torch.cat([
        #     torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
        #               dim=1) for u in x
        # ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            t = pnp_repeat(t, repeats=seq_lens)
            e = self.time_embedding(
                wan.sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # guidance embeddings
        if self.guidance_embeds:
            assert guidance is not None, f"guidance embeddings required for guidance_embeds=True"
            guidance = guidance * 1000
            with amp.autocast(dtype=torch.float32):
                guidance = pnp_repeat(guidance, repeats=seq_lens)
                g = self.guidance_embedding(
                    wan.sinusoidal_embedding_1d(self.freq_dim, guidance).float())
                g0 = self.guidance_projection(g).unflatten(1, (6, self.dim))
                assert g.dtype == torch.float32 and g0.dtype == torch.float32
                e0 = e0 + g0

        # context
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        context_lens = torch.tensor(
            [self.text_len for _ in context], dtype=torch.int32, device=device)

        # if clip_fea is not None:
        if self.model_type=="i2v":
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context_lens_img = torch.tensor(
                [context_clip.size(1) for _ in context_clip], dtype=torch.int32, device=device)
            context = [context, context_clip]
            context_lens = [context_lens, context_lens_img]

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        # for block in self.blocks:
        #     x = maybe_checkpoint(block, x, enabled=self.gradient_checkpointing, **kwargs)
        x = maybe_checkpoint(
            self.blocks,
            x,
            enabled=self.gradient_checkpointing and self.block_checkpoint_enabled,
            gc_step=self.block_checkpoint_step,
            gc_start_idx=self.block_checkpoint_start_idx,
            **kwargs
        )

        # head
        x = self.head(x, e)

        # unpatchify
        xs = x.split(seq_lens.tolist(), dim=0)
        x = self.unpatchify(xs, grid_sizes)
        return [u.float() for u in x]
