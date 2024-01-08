import os
import gc
import psutil
import numpy as np
import math
import jax
import jax.scipy as scipy
import jax.numpy as jnp
import jax.profiler
import flax
from flax import jax_utils
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import einops
import dm_pix as pix
from datasets import load_dataset, Image
import pyarrow as pa
import ml_collections
from ml_collections.config_dict import FrozenConfigDict, ConfigDict, create
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from absl import flags, app, logging
from tqdm import tqdm
import yaml
from typing import Sequence, Optional, Any, Callable, Sequence, List, Tuple, Union


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def l2norm(t, axis=1, eps=1e-12):
    denom = jnp.clip(jnp.linalg.norm(t, ord=2, axis=axis, keepdims=True), eps)
    out = t / denom
    return out


def mish(x: jnp.array):
    '''
    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    '''
    return x * jnp.tanh(jax.nn.softplus(x))


class Residual(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class PreNorm(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x) -> Any:
        x = nn.GroupNorm(1)(x)
        return self.fn(x)


class Downsample(nn.Module):
    dim: Optional[int]
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x) -> Any:
        B, H, W, C = x.shape
        dim = self.dim if self.dim is not None else C
        x = nn.Conv(
            dim, kernel_size=(4, 4), strides=(2, 2), padding=1, dtype=self.dtype
        )(x)
        assert x.shape == (B, H // 2, W // 2, dim)
        return x


class Upsample(nn.Module):
    dim: Optional[int]
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        dim = self.dim if self.dim is not None else C
        # x = jax.image.resize(x, (B, H * 2, W * 2, C), "nearest")
        x = x.repeat(2, axis=1).repeat(2, axis=2)
        x = nn.Conv(dim, kernel_size=(3, 3), padding=1, dtype=self.dtype)(x)
        assert x.shape == (B, H * 2, W * 2, dim)
        return x


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) * (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=self.dtype) * -emb)
        emb = t.astype(self.dtype)[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class WeightStandardizedConv(nn.Module):
    """
    apply weight standardization  https://arxiv.org/abs/1903.10520
    """

    features: int
    kernel_size: Sequence[int] = (3, 3)
    strides: Union[None, int, Sequence[int]] = 1
    padding: Any = 1
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)

        conv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        kernel_init = lambda rng, x: conv.init(rng, x)["params"]["kernel"]
        bias_init = lambda rng, x: conv.init(rng, x)["params"]["bias"]

        # standardize kernel
        kernel = self.param("kernel", kernel_init, x)
        eps = 1e-5 if self.dtype == jnp.float32 else 1e-3
        # reduce over dim_out
        redux = tuple(range(kernel.ndim - 1))
        mean = jnp.mean(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        var = jnp.var(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        standardized_kernel = (kernel - mean) / jnp.sqrt(var + eps)

        bias = self.param("bias", bias_init, x)

        return conv.apply({"params": {"kernel": standardized_kernel, "bias": bias}}, x)


class Block(nn.Module):
    dim: int = None
    groups: Optional[int] = 8
    kernel_size: Optional[int] = 3
    padding: Optional[int] = 1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, scale_shift):
        h = WeightStandardizedConv(
            features=self.dim,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=self.padding,
            name="conv_0",
        )(x)
        h = nn.GroupNorm(num_groups=self.groups, dtype=self.dtype, name="norm_0")(h)
        scale, shift = scale_shift
        h = h * (scale + 1) + shift
        return mish(h)


class ResNetBlock(nn.Module):
    dim: int = None
    groups: Optional[int] = 8
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, t_emb) -> Any:
        B, H, W, C = x.shape
        assert t_emb.shape[0] == B and len(t_emb.shape) == 2
        # add in positional embedding
        t_emb = nn.Dense(
            features=2 * self.dim, dtype=self.dtype, name="time_mlp.dense_0"
        )(mish(t_emb))
        t_emb = t_emb[
            :, jnp.newaxis, jnp.newaxis, :
        ]  # apply embedding across image batches and channels
        scale_shift = jnp.split(t_emb, 2, axis=-1)

        h = Block(self.dim, groups=self.groups, dtype=self.dtype)(x, scale_shift)
        h = Block(self.dim, groups=self.groups, dtype=self.dtype)(h, (0, 0))
        x = h + (
            nn.Conv(
                features=self.dim, kernel_size=(1, 1), dtype=self.dtype, name="res_conv"
            )(x)
            if C != self.dim
            else x
        )
        return x


class ConditionalResNetBlock(nn.Module):
    dim: int = None
    cond_dim: int = None
    cond_predict_scale: bool = False
    kernel_size: Optional[int] = 3
    groups: Optional[int] = 8

    @nn.compact
    def __call__(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = Block(
            self.dim, kernel_size=self.kernel_size, padding=0, groups=self.groups
        )(x, scale_shift=(0, 0))

        # cond encoder
        embed = nn.Sequential(
            [
                mish,
                nn.Linear(self.cond_dim, self.dim),
                lambda x: einops.rearrange(x, "b t -> b t 1"),
            ]
        )(cond)

        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...] # a
            bias = embed[:, 1, ...] # b 
            out = scale * out + bias
        else:
            out = out + embed
        out = Block(
            self.dim, kernel_size=self.kernel_size, padding=0, groups=self.groups
        )(x, scale_shift=(0, 0))
        # Residual conv
        out = out + nn.Conv(self.dim, kernel_size=(1, 1))(x)
        return out

class Attention(nn.Module):
    heads: int = 4
    dim_head: int = 32
    scale: int = 10
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        dim = self.dim_head * self.heads
        q, k, v = jnp.split(
            nn.Conv(dim * 3, kernel_size=(1, 1), use_bias=False)(x), 3, axis=-1
        )
        # split into heads
        q, k, v = map(
            lambda t: einops.rearrange(t, "b x y (h d) -> b (x y) h d", h=self.heads),
            (q, k, v),
        )

        assert q.shape == k.shape == v.shape == (B, H * W, self.heads, self.dim_head)
        # normalize
        q, k = map(l2norm, (q, k))
        # query-key scaled dot product attention
        attn = jnp.einsum("b i h d, b j h d -> b h i j", q, k) * self.scale
        attn = attn - attn.argmax(axis=-1, keepdims=True)
        attn = nn.softmax(attn, axis=-1)
        assert attn.shape == (B, self.heads, H * W, H * W)
        out = jnp.einsum("b h i j, b j h d  -> b h i d", attn, v)
        out = einops.rearrange(out, "b h (x y) d -> b x y (h d)", x=H)
        assert out.shape == (B, H, W, dim)
        out = nn.Conv(features=C, kernel_size=(1, 1))(out)
        return out

class LinearAttention(nn.Module):
    heads: int = 4
    dim_head: int = 32
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        dim = self.dim_head * self.heads
        q, k, v = jnp.split(
            nn.Conv(
                dim * 3, kernel_size=(1, 1), use_bias=False, name="lin_attn.conv_0"
            )(x),
            3,
            axis=-1,
        )
        q, k, v = map(
            lambda t: einops.rearrange(t, "b x y (h d) -> b (x y) h d", h=self.heads),
            (q, k, v),
        )
        q = nn.softmax(q, axis=-1) / jnp.sqrt(self.dim_head)
        k = nn.softmax(k, axis=1)
        v = v / (H * W)
        context = jnp.einsum("b n h d, b n h e -> b h d e", k, v)
        out = jnp.einsum("b h d e, b n h d -> b h e n", context, q)
        out = einops.rearrange(out, "b h e (x y) -> b x y (h e)", x=H)
        out = nn.Conv(
            features=C, kernel_size=(1, 1), dtype=self.dtype, name="lin_attn.conv_1"
        )(out)
        out = nn.LayerNorm(
            epsilon=1e-5, use_bias=False, dtype=self.dtype, name="lin_attn.norm_0"
        )(out)
        return out


class AttnBlock(nn.Module):
    """Convenience wrapper for Attention or LinearAttention"""

    heads: int = 4
    dim_head: int = 32
    use_linear_attention: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        norm_x = nn.LayerNorm(epsilon=1e-5, use_bias=False, dtype=self.dtype)(x)
        if self.use_linear_attention:
            attn = LinearAttention(self.heads, self.dim_head, dtype=self.dtype)
        else:
            attn = Attention(self.heads, self.dim_head, dtype=self.dtype)

        out = attn(norm_x)
        assert out.shape == x.shape
        return out + x




class UNet(nn.Module):
    dim: int
    init_dim: Optional[int] = None  # if None, same as dim
    out_dim: Optional[int] = None
    dim_mults: Tuple[int, int, int, int] = (1, 2, 4, 8)
    channels: int = 3
    resnet_block_groups: int = 8
    learned_variance: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, t) -> Any:
        B, H, W, C = x.shape
        init_dim = self.dim if self.init_dim is None else self.init_dim
        hs = []
        h = nn.Conv(
            features=init_dim,
            kernel_size=(7, 7),
            padding=3,
            name="init.conv_0",
            dtype=self.dtype,
        )(x)
        hs.append(h)
        # use sinusoidal embeddings to encode timesteps
        time_emb = SinusoidalPositionEmbeddings(self.dim, dtype=self.dtype)(
            t
        )  # [B. dim]
        time_emb = nn.Dense(
            features=self.dim * 4, dtype=self.dtype, name="time_mlp.dense_0"
        )(time_emb)
        time_emb = nn.Dense(
            features=self.dim * 4, dtype=self.dtype, name="time_mlp.dense_1"
        )(
            nn.gelu(time_emb)
        )  # [B, 4*dim]

        # downsampling
        num_resolutions = len(self.dim_mults)
        for i in range(num_resolutions):
            dim_in = h.shape[-1]
            h = ResNetBlock(
                dim=dim_in,
                groups=self.resnet_block_groups,
                dtype=self.dtype,
                name=f"down_{i}.resblock_0",
            )(h, time_emb)
            hs.append(h)
            h = ResNetBlock(
                dim=dim_in,
                groups=self.resnet_block_groups,
                dtype=self.dtype,
                name=f"down_{i}.resblock_1",
            )(h, time_emb)
            h = AttnBlock(dtype=self.dtype, name=f"down_{i}.attnblock_0")(h)
            hs.append(h)
            if i < num_resolutions - 1:
                h = Downsample(
                    dim=self.dim * self.dim_mults[i],
                    dtype=self.dtype,
                    name=f"down_{i}.downsample_0",
                )(h)

        mid_dim = self.dim * self.dim_mults[-1]
        h = nn.Conv(
            features=mid_dim,
            kernel_size=(3, 3),
            padding=1,
            dtype=self.dtype,
            name=f"down{num_resolutions-1}.conv_0",
        )(h)

        # middle
        h = ResNetBlock(
            dim=mid_dim,
            groups=self.resnet_block_groups,
            dtype=self.dtype,
            name="mid.resblock_0",
        )(h, time_emb)
        h = AttnBlock(
            use_linear_attention=False, dtype=self.dtype, name="mid.attnblock_0"
        )(h)
        h = ResNetBlock(
            dim=mid_dim,
            groups=self.resnet_block_groups,
            dtype=self.dtype,
            name="mid.resblock_1",
        )(h, time_emb)

        # upsampling
        for i in reversed(range(num_resolutions)):
            dim_in = self.dim * self.dim_mults[i]
            dim_out = self.dim * self.dim_mults[i - 1] if i > 0 else init_dim
            h = jnp.concatenate((h, hs.pop()), axis=-1)
            h = ResNetBlock(
                dim=dim_in,
                groups=self.resnet_block_groups,
                dtype=self.dtype,
                name=f"up_{i}.resblock_0",
            )(h, time_emb)
            h = jnp.concatenate((h, hs.pop()), axis=-1)
            h = ResNetBlock(
                dim=dim_in,
                groups=self.resnet_block_groups,
                dtype=self.dtype,
                name=f"up_{i}.resblock_1",
            )(h, time_emb)
            h = AttnBlock(dtype=self.dtype, name=f"up_{i}.attnblock_0")(h)
            if i > 0:
                h = Upsample(
                    dim=dim_out * self.dim_mults[i],
                    dtype=self.dtype,
                    name=f"up_{i}.downsample_0",
                )(h)

        h = jnp.concatenate((h, hs.pop()), axis=-1)
        out = ResNetBlock(
            dim=self.dim,
            groups=self.resnet_block_groups,
            dtype=self.dtype,
            name="final.resblock_0",
        )(h, time_emb)

        default_out_dim = C * (1 if not self.learned_variance else 2)
        out_dim = default_out_dim if self.out_dim is None else self.out_dim
        return nn.Conv(
            out_dim, kernel_size=(1, 1), dtype=self.dtype, name="final.conv_0"
        )(out)

class ConditionalUnet(nn.Module):
    dim: int
    init_dim: Optional[int] = None  # if None, same as dim
    out_dim: Optional[int] = None
    local_cond_dim: Optional[int] = None
    global_cond_dim: Optional[int] = None
    dim_mults: Tuple[int, int, int, int] = (1, 2, 4, 8)
    channels: int = 3
    resnet_block_groups: int = 8
    learned_variance: bool = False
    cond_predict_scale: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, t, local_cond=None, global_cond=None) -> Any:
        raise NotImplemented

## Training


def l1_loss(logit, target):
    return jnp.abs(logit - target)


def cosine_beta_schedule(timesteps):
    """Return cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672"""
    s = 0.008
    max_beta = 0.999
    ts = np.linspace(0, 1, timesteps + 1)
    alphas_bar = np.cos((ts + s) / (1 + s) * np.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return np.clip(betas, 0, max_beta)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = np.linspace(-6, 6, timesteps)
    return nn.sigmoid(betas) * (beta_end - beta_start) + beta_start


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = jnp.take(a, t, axis=-1)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def create_samplers(timesteps=200):
    # define beta schedule
    betas = cosine_beta_schedule(timesteps=timesteps)

    # define alphas
    alphas = 1.0 - betas
    # Needs to be on cpu because of jax-metal bug
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod_prev = np.pad(
        alphas_cumprod[:-1], pad_width=(1, 0), mode="constant", constant_values=1.0
    )
    sqrt_recip_alphas = np.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    # forward diffusion (using the nice property)
    def q_sample(x_start, t, noise=None):
        if noise is None:
            noise = jax.random.normal(jax.random.PRNGKey(0), shape=x_start.shape)

        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(model, x, t, t_index):
        betas_t = extract(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, x.shape)
            noise = jax.random.normal(jax.random.PRNGKey(0), shape=x.shape)
            return model_mean + jnp.sqrt(posterior_variance_t) * noise

    return q_sample, p_sample


def create_loss_fn(model: nn.Module, timesteps=200, loss_type="l1"):
    q_sample, _ = create_samplers(timesteps)

    if loss_type == "l1":
        loss_fn = l1_loss
    elif loss_type == "l2":
        loss_fn = optax.l2_loss
    elif loss_type == "huber":
        loss_fn = optax.huber_loss
    else:
        raise NotImplementedError()

    @jax.jit
    # @jax.vmap
    def p_losses(
        params,
        x_start,
        t,
        noise=None,
    ):
        if noise is None:
            noise = jax.random.normal(jax.random.PRNGKey(0), shape=x_start.shape)
        x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

        predicted_noise = model.apply(params, x_noisy, t)
        loss = loss_fn(predicted_noise, x_noisy).sum(axis=0)
        # assert jnp.isscalar(loss), f"shape: {loss.shape}"
        return loss.mean()

    return jax.value_and_grad(p_losses)


def create_transform(image_size: Union[int, Tuple[int, int]] = 256, dtype=jnp.float32):
    if type(image_size) == int:
        new_w, new_h = image_size, image_size
    else:
        new_w, new_h = image_size

    # @jax.vmap
    def transform(im):
        im = jnp.asarray(im)
        if len(im.shape) == 2:
            im = im[:, jnp.newaxis]
        w, h, c = im.shape
        # im = im.astype(dtype)
        im = jax.image.resize(im, (new_w, new_h, c), "linear")
        im = pix.resize_with_crop_or_pad(im, new_w, new_h)
        im = im / 255
        im = im * 2 - 1
        return im

    # @jax.vmap
    def reverse_transform(im):
        # im = im.squeeze()
        im = (im + 1) / 2
        im = im * 255
        im = im.astype(jnp.uint8)
        return im

    return transform, reverse_transform


def get_dataset(rng, config: FrozenConfigDict):
    if config.data.batch_size % jax.device_count() > 0:
        raise ValueError("Batch size must be divisible by the number of devices")
    batch_size = config.data.batch_size // jax.process_count()
    if config.training.half_precision:
        input_dtype = jnp.float32
    else:
        input_dtype = jnp.float16
    transform, _ = create_transform(config.data.image_size)

    def preprocess(batch: dict):
        batch["pixel_values"] = jnp.stack(
            [transform(jnp.asarray(im)) for im in batch["image"]], axis=0
        )
        # batch['image'] = transform(batch['image'])
        return batch

    dataset = load_dataset(
        config.data.hf_dataset,
        split="train",
        num_proc=8,
        keep_in_memory=True,
        streaming=False,
    )
    # dataset.set_format("jax")
    # dataset = dataset.map(preprocess)
    dataset = dataset.with_transform(preprocess)
    dataset.shuffle()
    # print(type(dataset[0]['image']))
    return dataset


def get_optimizers(config: ml_collections.FrozenConfigDict):
    raise NotImplemented


def train(config: ml_collections.FrozenConfigDict, workdir: str):
    writer = SummaryWriter(workdir)
    # TODO: set up tensorboard logging
    sample_dir = os.path.join(workdir, "samples")

    rng = jax.random.PRNGKey(config.seed)
    rng, d_rng = jax.random.split(rng)
    timesteps = config.ddpm.timesteps
    batch_size = config.data.batch_size

    dataset = get_dataset(rng, config)
    model = UNet(
        config.model.dim,
        channels=config.data.channels,
        dim_mults=config.model.dim_mults,
    )
    input_shape = (
        config.data.batch_size,
        config.data.image_size,
        config.data.image_size,
        config.data.channels,
    )
    x_0 = jax.random.uniform(rng, shape=input_shape)
    t_0 = jax.random.uniform(rng, shape=(config.data.batch_size,))
    params = model.init(rng, x_0, t_0)
    logging.info("model initialized!")
    ckpt_dir = os.path.join(workdir, "checkpoints")

    optim = optax.adam(learning_rate=float(config.optim.lr))
    opt_state = optim.init(params)
    p_loss_grad = create_loss_fn(model, timesteps, loss_type="huber")

    epochs = 5
    training_loss = []
    for epoch in range(epochs):
        pbar = tqdm(range(config.training.num_train_steps))
        for step, batch in zip(pbar, dataset.iter(batch_size)):
            gc.collect()
            batch = batch["pixel_values"]
            # batch = batch['image']
            t = jax.random.randint(
                rng, shape=(batch_size,), dtype=jnp.int32, minval=0, maxval=timesteps
            )
            loss, grad = p_loss_grad(params, batch, t)

            if step % config.training.log_every_steps == 0:
                logging.info("Loss: %f", loss.item())
                writer.add_scalar('loss/train', loss.item(), step)
                # jax.profiler.save_device_memory_profile(f"memory_{step}.prof")
            updates, opt_state = optim.update(grad, opt_state)
            params = optax.apply_updates(params, updates)

        if config.logging.save_checkpoint:
            logging.info("Saving checkpoint to: %s", ckpt_dir)
            checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=params, step=epoch)

def sample(config, workdir=None, wandb_artifact=None):
    raise NotImplementedError


########################################################
# Application

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config",
    None,
    "File path to the training or sampling hyperparameter configuration.",
)
flags.DEFINE_string("workdir", None, "Work unit directory.")

flags.DEFINE_boolean("debug", False, "Produces debugging output.")
flags.DEFINE_enum("mode", "train", ["train", "sample"], "Running mode: train or sample")


def yaml_to_config(yaml_config_path: str) -> FrozenConfigDict:
    config = FrozenConfigDict(yaml.load(open(FLAGS.config).read(), yaml.CLoader))
    # print(config)
    return config


def main(argv):

    logging.info("JAX process: %d / %d", jax.process_index() + 1, jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())
    if FLAGS.debug:
        logging.debug("debug mode")
    if FLAGS.mode == "train":
        train(yaml_to_config(FLAGS.config), FLAGS.workdir)
    elif FLAGS.mode == "sample":
        sample(yaml_to_config(FLAGS.config))
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
