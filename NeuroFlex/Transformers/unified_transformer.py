import torch
from torch import nn as torch_nn
import math
import jax
import jax.numpy as jnp
import flax.linen as flax_nn
import haiku as hk
import sonnet as snt
import tensorflow as tf
from jax import random
from typing import Any, Callable

class FrameworkWrapper:
    def __init__(self, backend='pytorch'):
        self.backend = backend

    def get_module(self, module_class):
        if self.backend == 'pytorch':
            return module_class
        elif self.backend == 'jax':
            return hk.to_module(module_class)
        elif self.backend == 'flax':
            return flax_nn.Module
        elif self.backend == 'sonnet':
            return snt.Module
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

framework = FrameworkWrapper()

class MultiHeadAttention(framework.get_module(torch_nn.Module)):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        if framework.backend == 'pytorch':
            self.wq = torch_nn.Linear(d_model, d_model)
            self.wk = torch_nn.Linear(d_model, d_model)
            self.wv = torch_nn.Linear(d_model, d_model)
            self.dense = torch_nn.Linear(d_model, d_model)
        elif framework.backend == 'jax':
            self.wq = hk.Linear(d_model)
            self.wk = hk.Linear(d_model)
            self.wv = hk.Linear(d_model)
            self.dense = hk.Linear(d_model)
        elif framework.backend == 'flax':
            self.wq = flax_nn.Dense(d_model)
            self.wk = flax_nn.Dense(d_model)
            self.wv = flax_nn.Dense(d_model)
            self.dense = flax_nn.Dense(d_model)
        elif framework.backend == 'sonnet':
            self.wq = snt.Linear(d_model)
            self.wk = snt.Linear(d_model)
            self.wv = snt.Linear(d_model)
            self.dense = snt.Linear(d_model)

    def split_heads(self, x, batch_size):
        if framework.backend == 'pytorch':
            return x.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        elif framework.backend in ['jax', 'flax']:
            return jnp.transpose(x.reshape(batch_size, -1, self.num_heads, self.depth), (0, 2, 1, 3))
        elif framework.backend == 'sonnet':
            return tf.transpose(tf.reshape(x, (batch_size, -1, self.num_heads, self.depth)), perm=[0, 2, 1, 3])

    def linear_attention(self, q, k, v, mask=None):
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        if mask is not None:
            # Ensure mask has the correct shape for broadcasting
            while mask.dim() < k.dim():
                mask = mask.unsqueeze(1)
            # Expand mask to match the batch size, number of heads, and sequence length
            mask = mask.expand(k.size(0), k.size(1), k.size(2), -1)
            # Expand mask to match the last dimension of k
            mask = mask.unsqueeze(-1).expand_as(k)
            # Apply the mask to k
            k = k * mask.to(k.dtype)

        context = k.transpose(-2, -1) @ v
        out = q @ context
        return out

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        if framework.backend == 'pytorch':
            output = self.linear_attention(q, k, v, mask)
        elif framework.backend in ['jax', 'flax']:
            output = jax.vmap(self.linear_attention)(q, k, v, mask)
        elif framework.backend == 'sonnet':
            output = tf.vectorized_map(lambda x: self.linear_attention(*x), (q, k, v, mask))

        if framework.backend == 'pytorch':
            output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        elif framework.backend in ['jax', 'flax']:
            output = jnp.transpose(output, (0, 2, 1, 3)).reshape(batch_size, -1, self.d_model)
        elif framework.backend == 'sonnet':
            output = tf.reshape(tf.transpose(output, perm=[0, 2, 1, 3]), (batch_size, -1, self.d_model))

        output = self.dense(output)

        return output, None  # Return None instead of attention weights for consistency

class PositionalEncoding(framework.get_module(torch_nn.Module)):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def _get_angles(self, pos, i):
        if framework.backend == 'pytorch':
            angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / self.d_model)
            return pos * angle_rates
        elif framework.backend in ['jax', 'flax']:
            angle_rates = 1 / jnp.power(10000, (2 * (i // 2)) / self.d_model)
            return pos * angle_rates
        elif framework.backend in ['sonnet', 'tensorflow']:
            angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(self.d_model, tf.float32))
            return pos * angle_rates

    def forward(self, x):
        if framework.backend == 'pytorch':
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
            div_term = torch.arange(0, self.d_model, 2, dtype=torch.float32)
            angles = self._get_angles(position, div_term)
            emb = torch.zeros(x.size(1), self.d_model).to(x.device)
            emb[:, 0::2] = torch.sin(angles)
            emb[:, 1::2] = torch.cos(angles)
            return x + emb.unsqueeze(0)
        elif framework.backend in ['jax', 'flax']:
            position = jnp.arange(0, x.shape[1], dtype=jnp.float32)[:, jnp.newaxis]
            div_term = jnp.arange(0, self.d_model, 2, dtype=jnp.float32)
            angles = self._get_angles(position, div_term)
            emb = jnp.zeros((x.shape[1], self.d_model))
            emb = emb.at[:, 0::2].set(jnp.sin(angles))
            emb = emb.at[:, 1::2].set(jnp.cos(angles))
            return x + emb[jnp.newaxis, :, :]
        elif framework.backend in ['sonnet', 'tensorflow']:
            position = tf.range(0, tf.shape(x)[1], dtype=tf.float32)[:, tf.newaxis]
            div_term = tf.range(0, self.d_model, 2, dtype=tf.float32)
            angles = self._get_angles(position, div_term)
            emb = tf.zeros((tf.shape(x)[1], self.d_model))
            emb = emb + tf.scatter_nd(tf.constant([[0, 1]]), tf.sin(angles), tf.shape(emb))
            emb = emb + tf.scatter_nd(tf.constant([[1, 0]]), tf.cos(angles), tf.shape(emb))
            return x + tf.expand_dims(emb, 0)

class FeedForward(framework.get_module(torch_nn.Module)):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        if framework.backend == 'pytorch':
            self.linear1 = torch_nn.Linear(d_model, d_ff)
            self.linear2 = torch_nn.Linear(d_ff, d_model)
        elif framework.backend == 'jax':
            self.linear1 = hk.Linear(d_ff)
            self.linear2 = hk.Linear(d_model)
        elif framework.backend == 'flax':
            self.linear1 = flax_nn.Dense(d_ff)
            self.linear2 = flax_nn.Dense(d_model)
        elif framework.backend == 'sonnet':
            self.linear1 = snt.Linear(d_ff)
            self.linear2 = snt.Linear(d_model)

    def forward(self, x):
        if framework.backend == 'pytorch':
            return self.linear2(torch.relu(self.linear1(x)))
        elif framework.backend in ['jax', 'flax']:
            return self.linear2(jax.nn.relu(self.linear1(x)))
        elif framework.backend == 'sonnet':
            return self.linear2(tf.nn.relu(self.linear1(x)))

class EncoderLayer(framework.get_module(torch_nn.Module)):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        if framework.backend == 'pytorch':
            self.layernorm1 = torch_nn.LayerNorm(d_model)
            self.layernorm2 = torch_nn.LayerNorm(d_model)
            self.dropout = torch_nn.Dropout(dropout)
        elif framework.backend in ['jax', 'flax']:
            self.layernorm1 = flax_nn.LayerNorm()
            self.layernorm2 = flax_nn.LayerNorm()
            self.dropout = lambda x: flax_nn.Dropout(rate=dropout)(x, deterministic=False)
        elif framework.backend == 'sonnet':
            self.layernorm1 = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            self.layernorm2 = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            self.dropout = lambda x: tf.nn.dropout(x, rate=dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        out1 = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout(ffn_output))
        return out2

class UnifiedTransformer(framework.get_module(torch_nn.Module)):
    """
    UnifiedTransformer: A transformer architecture that combines elements from BERT, GPT, LLaMA, and T5 models.

    This class implements a unified transformer that incorporates:
    1. Bidirectional encoding (BERT-like) for understanding context in both directions.
       (Devlin et al., 2018: https://arxiv.org/abs/1810.04805)
    2. Autoregressive decoding (GPT-like) for text generation.
       (Brown et al., 2020: https://arxiv.org/abs/2005.14165)
    3. Efficient training techniques inspired by LLaMA for handling large datasets.
       (Touvron et al., 2023: https://arxiv.org/abs/2302.13971)
    4. Text-to-text framework (T5-like) for versatile task adaptation.
       (Raffel et al., 2020: https://arxiv.org/abs/1910.10683)

    The architecture aims to leverage the strengths of each model type while providing a flexible
    framework for various NLP tasks.
    """

    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=512, dropout=0.1):
        """
        Initialize the UnifiedTransformer.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the model (default: 512).
            num_heads (int): Number of attention heads (default: 8).
            num_layers (int): Number of transformer layers (default: 6).
            d_ff (int): Dimension of the feed-forward network (default: 2048).
            max_seq_length (int): Maximum sequence length (default: 512).
            dropout (float): Dropout rate (default: 0.1).
        """
        super(UnifiedTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = torch_nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        # Encoder layers for bidirectional context understanding (BERT-like)
        self.encoder_layers = torch_nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # Decoder layers for autoregressive generation (GPT-like)
        self.decoder_layers = torch_nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = torch_nn.Dropout(dropout)
        self.final_layer_norm = torch_nn.LayerNorm(d_model)
        # Language model head for text generation and task-specific outputs
        self.lm_head = torch_nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, mask=None, decoder_input_ids=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.long)
        else:
            x = x.long()

        x = torch.clamp(x, 0, self.vocab_size - 1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Bidirectional encoding (BERT-like)
        for layer in self.encoder_layers:
            if mask is not None:
                # Ensure mask has the correct shape
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1).unsqueeze(2)
                elif mask.dim() == 3:
                    mask = mask.unsqueeze(1)
            x = layer(x, mask)

        # Decoder for text generation (GPT-like)
        if decoder_input_ids is not None:
            if not isinstance(decoder_input_ids, torch.Tensor):
                decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long)
            else:
                decoder_input_ids = decoder_input_ids.long()

            decoder_input_ids = torch.clamp(decoder_input_ids, 0, self.vocab_size - 1)
            decoder_input = self.embedding(decoder_input_ids) * math.sqrt(self.d_model)
            decoder_input = self.pos_encoding(decoder_input)
            for layer in self.decoder_layers:
                if mask is not None:
                    # Ensure mask has the correct shape for decoder
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(1).unsqueeze(2)
                    elif mask.dim() == 3:
                        mask = mask.unsqueeze(1)
                decoder_input = layer(decoder_input, mask)
            x = decoder_input

        x = self.final_layer_norm(x)
        return x

    def generate(self, input_ids, max_length, temperature=1.0, top_k=50):
        """
        Text generation with efficiency improvements inspired by LLaMA.

        This method implements autoregressive text generation with top-k sampling
        and temperature scaling for controlling output diversity.

        Args:
            input_ids (torch.Tensor): Input token ids.
            max_length (int): Maximum length of the generated sequence.
            temperature (float): Temperature for controlling randomness in sampling.
            top_k (int): Number of top tokens to consider for sampling.

        Returns:
            torch.Tensor: Generated token ids.
        """
        generated = input_ids.clamp(0, self.vocab_size - 1)
        while generated.size(1) < max_length:
            outputs = self.forward(generated)
            next_token_logits = self.lm_head(outputs[:, -1, :])
            next_token_logits = next_token_logits / temperature

            # Top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k=min(top_k, self.vocab_size), dim=-1)
            next_token_probs = torch.softmax(top_k_logits, dim=-1)

            # Sample from the filtered distribution
            next_token_index = torch.multinomial(next_token_probs, num_samples=1)
            next_token = top_k_indices.gather(-1, next_token_index)

            generated = torch.cat([generated, next_token], dim=-1)

        return generated[:, :max_length]

    def text_to_text(self, input_ids, target_ids):
        """
        Text-to-text framework inspired by T5 for versatile task adaptation.

        This method allows the model to be fine-tuned for various NLP tasks
        by framing them as text-to-text problems.

        Args:
            input_ids (torch.Tensor or jnp.ndarray): Input token ids.
            target_ids (torch.Tensor or jnp.ndarray): Target token ids.

        Returns:
            torch.Tensor or jnp.ndarray: Logits for the target sequence.
        """
        if framework.backend == 'pytorch':
            return self._text_to_text_pytorch(input_ids, target_ids)
        elif framework.backend in ['jax', 'flax']:
            return self._text_to_text_jax(input_ids, target_ids)
        else:
            raise ValueError(f"Unsupported backend: {framework.backend}")

    def _text_to_text_pytorch(self, input_ids, target_ids):
        torch.manual_seed(42)  # Set a fixed seed for deterministic behavior
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        target_ids = torch.clamp(target_ids, 0, self.vocab_size - 1)

        try:
            encoder_output = self.forward(input_ids)
            if target_ids.dim() == 3:
                target_ids = target_ids.squeeze(-1)
            if encoder_output.size(1) != target_ids.size(1):
                encoder_output = encoder_output[:, :target_ids.size(1), :]

            decoder_input_ids = torch.cat([torch.zeros_like(target_ids[:, :1]), target_ids[:, :-1]], dim=1)
            decoder_input = self.embedding(decoder_input_ids) * math.sqrt(self.d_model)
            decoder_input = self.pos_encoding(decoder_input)

            for layer in self.decoder_layers:
                decoder_input = layer(decoder_input)

            decoder_output = self.final_layer_norm(decoder_input)
            lm_logits = self.lm_head(decoder_output)
        except RuntimeError as e:
            print(f"Error in text_to_text_pytorch: {e}")
            return None

        return lm_logits

    def _text_to_text_jax(self, input_ids, target_ids):
        key = jax.random.PRNGKey(42)  # Set a fixed seed for deterministic behavior
        input_ids = jnp.clip(input_ids, 0, self.vocab_size - 1)
        target_ids = jnp.clip(target_ids, 0, self.vocab_size - 1)

        try:
            encoder_output = self.forward(input_ids)
            if len(target_ids.shape) == 3:
                target_ids = jnp.squeeze(target_ids, axis=-1)
            if encoder_output.shape[1] != target_ids.shape[1]:
                encoder_output = encoder_output[:, :target_ids.shape[1], :]

            decoder_input_ids = jnp.concatenate([jnp.zeros_like(target_ids[:, :1]), target_ids[:, :-1]], axis=1)
            decoder_input = self.embedding(decoder_input_ids) * jnp.sqrt(self.d_model)
            decoder_input = self.pos_encoding(decoder_input)

            for layer in self.decoder_layers:
                decoder_input = layer(decoder_input)

            decoder_output = self.final_layer_norm(decoder_input)
            lm_logits = self.lm_head(decoder_output)
        except Exception as e:
            print(f"Error in text_to_text_jax: {e}")
            return None

        return lm_logits

    def few_shot_learning(self, support_set, query):
        """
        Few-shot learning capabilities inspired by GPT-3.

        This method allows the model to perform few-shot learning by providing
        a support set of examples and a query.

        Args:
            support_set (list): List of support set examples.
            query (torch.Tensor): Query input.

        Returns:
            torch.Tensor: Probability distribution over the vocabulary for the query.
        """
        if not support_set:
            raise ValueError("Support set cannot be empty")
        context = torch.cat([torch.clamp(s, 0, self.vocab_size - 1) for s in support_set], dim=1)
        query = torch.clamp(query, 0, self.vocab_size - 1)
        full_input = torch.cat([context, query], dim=1)
        output = self.forward(full_input)
        logits = self.lm_head(output[:, -query.size(1):, :])
        return torch.softmax(logits, dim=-1)  # Return probabilities for consistency

    def fine_tune(self, task='classification', num_labels=2):
        """
        Fine-tune the model for a specific task.

        This method adapts the model for either classification or generation tasks.

        Args:
            task (str): The task to fine-tune for ('classification' or 'generation').
            num_labels (int): Number of labels for classification task.
        """
        if task == 'classification':
            if framework.backend == 'pytorch':
                self.task_head = torch_nn.Linear(self.d_model, num_labels)
            elif framework.backend == 'jax':
                self.task_head = hk.Linear(num_labels)
            elif framework.backend == 'flax':
                self.task_head = flax_nn.Dense(num_labels)
            elif framework.backend == 'sonnet':
                self.task_head = snt.Linear(num_labels)
            else:
                raise ValueError(f"Unsupported backend: {framework.backend}")
        elif task == 'generation':
            self.task_head = self.lm_head
        else:
            raise ValueError(f"Unsupported task: {task}")

    def task_specific_forward(self, input_ids, attention_mask=None, task='classification'):
        """
        Perform a task-specific forward pass.

        This method allows the model to be used for different tasks after fine-tuning.

        Args:
            input_ids (torch.Tensor): Input token ids.
            attention_mask (torch.Tensor): Attention mask.
            task (str): The task to perform ('classification' or 'generation').

        Returns:
            torch.Tensor: Task-specific output (logits for classification or generation).
        """
        output = self.forward(input_ids, attention_mask)
        if task == 'classification':
            return self.task_head(output[:, 0, :])  # Use first token for classification
        elif task == 'generation':
            return self.task_head(output)  # Use all tokens for generation
        else:
            raise ValueError(f"Unsupported task: {task}")

def get_unified_transformer(backend='pytorch', vocab_size=30000):
    if backend == 'pytorch':
        return UnifiedTransformer(vocab_size)
    elif backend == 'jax':
        return JAXUnifiedTransformer(vocab_size)
    elif backend == 'flax':
        return FlaxUnifiedTransformer(vocab_size)
    elif backend == 'sonnet':
        return SonnetUnifiedTransformer(vocab_size)
    elif backend == 'tensorflow':
        raise NotImplementedError("TensorFlow backend not implemented yet")
    else:
        raise ValueError(f"Unsupported backend: {backend}")

class JAXUnifiedTransformer(UnifiedTransformer):
    def __init__(self, vocab_size):
        super().__init__(vocab_size)
        # JAX-specific initialization

class FlaxUnifiedTransformer(flax_nn.Module):
    vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    max_seq_length: int = 512
    dropout: float = 0.1

    def setup(self):
        self.embedding = flax_nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
        self.encoder_layers = [EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout) for _ in range(self.num_layers)]
        self.decoder_layers = [EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout) for _ in range(self.num_layers)]
        self.final_layer_norm = flax_nn.LayerNorm(epsilon=1e-5)
        self.lm_head = flax_nn.Dense(self.vocab_size)

    def __call__(self, x, mask=None, decoder_input_ids=None):
        x = self.embedding(x) * jnp.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = flax_nn.Dropout(rate=self.dropout)(x, deterministic=False)

        for layer in self.encoder_layers:
            x = layer(x, mask)

        if decoder_input_ids is not None:
            decoder_input = self.embedding(decoder_input_ids) * jnp.sqrt(self.d_model)
            decoder_input = self.pos_encoding(decoder_input)
            for layer in self.decoder_layers:
                decoder_input = layer(decoder_input, mask)
            x = decoder_input

        x = self.final_layer_norm(x)
        return x

class SonnetUnifiedTransformer(snt.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.dropout = dropout

        self.embedding = snt.Embed(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        self.decoder_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        self.final_layer_norm = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.lm_head = snt.Linear(vocab_size)

    def __call__(self, x, mask=None, decoder_input_ids=None):
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = tf.nn.dropout(x, rate=self.dropout)

        for layer in self.encoder_layers:
            x = layer(x, mask)

        if decoder_input_ids is not None:
            decoder_input = self.embedding(decoder_input_ids) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            decoder_input = self.pos_encoding(decoder_input)
            for layer in self.decoder_layers:
                decoder_input = layer(decoder_input, mask)
            x = decoder_input

        x = self.final_layer_norm(x)
        return x
