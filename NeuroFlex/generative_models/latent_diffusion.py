import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable, Dict, Any
import optax
from transformers import FlaxCLIPTextModel, CLIPTokenizer, FlaxCLIPModel, FlaxCLIPVisionModel

def sinusoidal_embedding(x):
    frequencies = jnp.exp(
        jnp.linspace(jnp.log(1.0), jnp.log(1000.0), 64)
    )
    angular_speeds = 2.0 * jnp.pi * frequencies
    embeddings = jnp.concatenate(
        [jnp.sin(angular_speeds * x), jnp.cos(angular_speeds * x)], axis=-1
    )
    return embeddings

def process_text(text: str, tokenizer: CLIPTokenizer, text_encoder: FlaxCLIPTextModel) -> jnp.ndarray:
    tokens = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="jax").input_ids
    text_embeddings = text_encoder(tokens)[0]
    return text_embeddings

class UNet(nn.Module):
    ch: int
    ch_mult: Tuple[int]
    num_res_blocks: int
    attn_resolutions: Tuple[int]
    dropout: float
    resamp_with_conv: bool

    @nn.compact
    def __call__(self, x, t, train):
        temb = sinusoidal_embedding(t)
        temb = nn.Dense(features=self.ch*4)(temb)
        temb = nn.relu(temb)
        temb = nn.Dense(features=self.ch*4)(temb)

        h = nn.Conv(features=self.ch, kernel_size=(3, 3), padding="SAME")(x)

        # Downsampling
        hs = [h]
        for i_level in range(len(self.ch_mult)):
            for i_block in range(self.num_res_blocks):
                h = self.ResnetBlock(out_ch=self.ch*self.ch_mult[i_level])(hs[-1], temb, train)
                if h.shape[1] in self.attn_resolutions:
                    h = self.AttnBlock()(h)
                hs.append(h)
            if i_level != len(self.ch_mult) - 1:
                h = self.Downsample()(hs[-1])
                hs.append(h)

        # Middle
        h = hs[-1]
        h = self.ResnetBlock(out_ch=self.ch*self.ch_mult[-1])(h, temb, train)
        h = self.AttnBlock()(h)
        h = self.ResnetBlock(out_ch=self.ch*self.ch_mult[-1])(h, temb, train)

        # Upsampling
        for i_level in reversed(range(len(self.ch_mult))):
            for i_block in range(self.num_res_blocks + 1):
                h = self.ResnetBlock(out_ch=self.ch*self.ch_mult[i_level])(
                    jnp.concatenate([h, hs.pop()], axis=-1), temb, train
                )
                if h.shape[1] in self.attn_resolutions:
                    h = self.AttnBlock()(h)
            if i_level != 0:
                h = self.Upsample()(h)

        h = nn.relu(nn.GroupNorm(num_groups=32)(h))
        h = nn.Conv(features=x.shape[-1], kernel_size=(3, 3), padding="SAME", kernel_init=nn.initializers.zeros)(h)
        return h

    class ResnetBlock(nn.Module):
        out_ch: int

        @nn.compact
        def __call__(self, x, temb, train):
            h = nn.relu(nn.GroupNorm(num_groups=32)(x))
            h = nn.Conv(features=self.out_ch, kernel_size=(3, 3), padding="SAME")(h)
            h += nn.Dense(features=self.out_ch)(nn.relu(temb))[:, None, None, :]
            h = nn.relu(nn.GroupNorm(num_groups=32)(h))
            h = nn.Conv(features=self.out_ch, kernel_size=(3, 3), padding="SAME", kernel_init=nn.initializers.zeros)(h)
            if x.shape[-1] != self.out_ch:
                x = nn.Conv(features=self.out_ch, kernel_size=(1, 1))(x)
            return x + h

    class AttnBlock(nn.Module):
        @nn.compact
        def __call__(self, x):
            h = nn.GroupNorm(num_groups=32)(x)
            q = nn.Conv(features=x.shape[-1], kernel_size=(1, 1))(h)
            k = nn.Conv(features=x.shape[-1], kernel_size=(1, 1))(h)
            v = nn.Conv(features=x.shape[-1], kernel_size=(1, 1))(h)

            w = jnp.einsum('bhwc,bHWc->bhwHW', q, k) * (int(x.shape[-1]) ** (-0.5))
            w = jnp.reshape(w, (*x.shape[:3], -1))
            w = jax.nn.softmax(w, axis=-1)
            w = jnp.reshape(w, (*x.shape[:3], *x.shape[1:3]))

            h = jnp.einsum('bhwHW,bHWc->bhwc', w, v)
            h = nn.Conv(features=x.shape[-1], kernel_size=(1, 1), kernel_init=nn.initializers.zeros)(h)
            return x + h

    class Downsample(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Conv(features=x.shape[-1], kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)

    class Upsample(nn.Module):
        @nn.compact
        def __call__(self, x):
            B, H, W, C = x.shape
            x = jax.image.resize(x, (B, H * 2, W * 2, C), method='nearest')
            return nn.Conv(features=C, kernel_size=(3, 3), padding="SAME")(x)

class VAE(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x, train):
        # Encoder
        h = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        h = nn.relu(h)
        h = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(h)
        h = nn.relu(h)
        h = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(h)
        h = nn.relu(h)
        h = h.reshape((h.shape[0], -1))
        h = nn.Dense(features=self.latent_dim * 2)(h)
        mean, logvar = jnp.split(h, 2, axis=-1)

        # Reparameterization trick
        if train:
            std = jnp.exp(0.5 * logvar)
            eps = jax.random.normal(self.make_rng('latent'), std.shape)
            z = mean + eps * std
        else:
            z = mean

        # Decoder
        h = nn.Dense(features=8*8*64)(z)
        h = h.reshape((-1, 8, 8, 64))
        h = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2))(h)
        h = nn.relu(h)
        h = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2))(h)
        h = nn.relu(h)
        h = nn.ConvTranspose(features=x.shape[-1], kernel_size=(3, 3), strides=(2, 2))(h)

        return h, mean, logvar

class LatentDiffusionModel(nn.Module):
    latent_dim: int
    image_size: Tuple[int, int, int]
    num_timesteps: int
    text_embed_dim: int
    context_dim: int
    clip_model_name: str = "openai/clip-vit-base-patch32"

    def setup(self):
        self.vae = VAE(latent_dim=self.latent_dim)
        self.unet = UNet(ch=128, ch_mult=(1, 2, 3, 4), num_res_blocks=2, attn_resolutions=(16,), dropout=0.1, resamp_with_conv=True)
        self.text_encoder = FlaxCLIPTextModel.from_pretrained(self.clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.clip_model_name)
        self.image_encoder = FlaxCLIPVisionModel.from_pretrained(self.clip_model_name)
        self.betas = jnp.linspace(1e-4, 0.02, self.num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)

    def __call__(self, x, t, train=True, text=None, image=None):
        if text is not None:
            embedding = self.text_to_latent(text)
        elif image is not None:
            embedding = self.image_to_latent(image)
        else:
            embedding = None

        z, _, _ = self.vae(x, train)
        return self.unet(z, t, embedding, train)

    def compute_loss(self, x, text, rng):
        rng, step_rng = jax.random.split(rng)
        t = jax.random.randint(step_rng, (x.shape[0],), 0, self.num_timesteps)
        z, mean, logvar = self.vae(x, True)

        text_embedding = self.text_to_latent(text)

        noise = jax.random.normal(rng, z.shape)
        noisy_z = jnp.sqrt(self.alphas_cumprod[t])[:, None, None, None] * z + \
                  jnp.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None] * noise

        predicted_noise = self.unet(noisy_z, t / self.num_timesteps, text_embedding, True)

        noise_loss = jnp.mean((noise - predicted_noise) ** 2)
        kl_loss = -0.5 * jnp.sum(1 + logvar - mean**2 - jnp.exp(logvar), axis=-1)

        image_embedding = self.image_to_latent(self.vae.decode(z))
        text_image_loss = optax.cosine_similarity(text_embedding, image_embedding).mean()

        total_loss = noise_loss + jnp.mean(kl_loss) - text_image_loss

        return total_loss

    def generate(self, rng, num_samples, input_type='random', input_data=None):
        if input_type == 'random':
            z = jax.random.normal(rng, (num_samples,) + self.image_size[:-1] + (self.latent_dim,))
        elif input_type == 'text':
            z = self.text_to_latent(input_data)
        elif input_type == 'image':
            z = self.image_to_latent(input_data)
        else:
            raise ValueError("Invalid input_type. Must be 'random', 'text', or 'image'.")

        def step(i, z):
            t = self.num_timesteps - i - 1
            t_emb = jnp.array([t / self.num_timesteps])
            noise_pred = self.unet(z, t_emb, None, train=False)
            z = self.p_sample(z, t, noise_pred)
            return z

        z = jax.lax.fori_loop(0, self.num_timesteps, step, z)
        return self.vae.decode(z)

    def p_sample(self, x, t, noise_pred):
        alpha_t = self.alphas[t]
        alpha_t_bar = self.alphas_cumprod[t]

        mean = (1 / jnp.sqrt(alpha_t)) * (x - ((1 - alpha_t) / jnp.sqrt(1 - alpha_t_bar)) * noise_pred)
        var = self.betas[t]

        noise = jax.random.normal(self.make_rng('sample'), x.shape)
        return mean + jnp.sqrt(var) * noise

    def encode(self, x):
        return self.vae.apply({'params': self.variables['params']['vae']}, x, False)[1]

    def decode(self, z):
        return self.vae.apply({'params': self.variables['params']['vae']}, z, False)[0]

    def text_to_latent(self, text):
        tokens = self.tokenizer(text, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="jax").input_ids
        text_embeddings = self.text_encoder(tokens, train=False)[0]
        return text_embeddings

    def image_to_latent(self, image):
        image_embeddings = self.image_encoder(image, train=False)[0]
        return image_embeddings
