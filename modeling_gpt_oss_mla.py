"""
GPT-OSS Multi-Head Latent Attention (MLA) Implementation

This module implements a true MLA architecture for GPT-OSS models, featuring:
- Compressed KV cache via learned projections
- RoPE applied after decompression  
- Preserved MoE and GQA structure
- Alternating sliding window / full attention layers

Based on DeepSeek-V2/V3 MLA architecture adapted for GPT-OSS.
"""

import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)


class GptOssMlaConfig(PretrainedConfig):
    """
    Configuration class for GPT-OSS MLA model.
    
    Extends the original GPT-OSS config with MLA-specific parameters.
    """
    model_type = "gpt_oss_mla"
    
    def __init__(
        self,
        vocab_size=201088,
        hidden_size=2880,
        intermediate_size=2880,
        num_hidden_layers=24,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        kv_lora_rank=384,  # MLA latent dimension (was 512 KV dims → 384 compressed)
        q_lora_rank=None,  # Optional Q compression
        rope_theta=150000,
        rope_scaling=None,
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=199999,
        bos_token_id=None,
        eos_token_id=200002,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        # MoE parameters
        num_local_experts=32,
        num_experts_per_tok=4,
        router_aux_loss_coef=0.9,
        # Layer types
        layer_types=None,  # List of "sliding_attention" or "full_attention"
        sliding_window=128,
        attention_bias=True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_aux_loss_coef = router_aux_loss_coef
        self.layer_types = layer_types or ["full_attention"] * num_hidden_layers
        self.sliding_window = sliding_window
        self.attention_bias = attention_bias
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )


class GptOssMlaRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class GptOssMlaRotaryEmbedding(nn.Module):
    """Rotary Position Embedding with YARN scaling support."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Args:
            x: [batch, num_heads, seq_len, head_dim]
            position_ids: [batch, seq_len]
        Returns:
            cos, sin: [batch, seq_len, head_dim]
        """
        # Expand position_ids
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Compute freqs
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embedding to queries and keys."""
    # q, k: [batch, num_heads, seq_len, head_dim]
    # cos, sin: [batch, seq_len, head_dim]
    cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GptOssMlaAttention(nn.Module):
    """
    Multi-Head Latent Attention with KV compression.
    
    Key differences from standard attention:
    1. KV projections compressed to latent_dim (kv_lora_rank)
    2. KV cache stores compressed representations
    3. Decompression happens before attention computation
    4. RoPE applied after decompression
    """
    
    def __init__(self, config: GptOssMlaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        
        # Query projection (standard)
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias
        )
        
        # MLA KV compression and decompression
        # Down projection: hidden -> latent (compress)
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank,
            bias=False
        )
        
        # Up projection: latent -> kv (decompress)
        # Output: n_kv_heads * head_dim * 2 (for K and V)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_key_value_heads * self.head_dim * 2,
            bias=False
        )
        
        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias
        )
        
        # RoPE
        self.rotary_emb = GptOssMlaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, cache_len + seq_len]
            position_ids: [batch, seq_len]
            past_key_value: Tuple of (compressed_kv,) from previous step
            
        Returns:
            attn_output: [batch, seq_len, hidden_size]
            attn_weights: Optional attention weights
            past_key_value: Tuple of (compressed_kv,) for caching
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Compute queries (standard)
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # MLA: Compress hidden states to latent KV
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [batch, seq_len, kv_lora_rank]
        
        # Handle KV cache
        if past_key_value is not None:
            # Concatenate with cached compressed KV
            compressed_kv = torch.cat([past_key_value[0], compressed_kv], dim=1)
        
        # MLA: Decompress latent to full KV
        kv = self.kv_b_proj(compressed_kv)  # [batch, cache_len + seq_len, num_kv_heads * head_dim * 2]
        kv = kv.view(batch_size, -1, self.num_key_value_heads, self.head_dim * 2)
        
        # Split into K and V
        key_states, value_states = kv.chunk(2, dim=-1)  # Each: [batch, cache_len + seq_len, num_kv_heads, head_dim]
        key_states = key_states.transpose(1, 2)  # [batch, num_kv_heads, cache_len + seq_len, head_dim]
        value_states = value_states.transpose(1, 2)
        
        # Compute full position_ids for cache
        kv_seq_len = key_states.shape[-2]
        if position_ids is None:
            position_ids = torch.arange(kv_seq_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0)
        elif past_key_value is not None:
            # Extend position IDs for cached tokens
            cache_len = past_key_value[0].shape[1]
            past_position_ids = torch.arange(cache_len, dtype=torch.long, device=hidden_states.device)
            position_ids = torch.cat([past_position_ids.unsqueeze(0).expand(batch_size, -1), position_ids], dim=1)
        
        # Apply RoPE (AFTER decompression)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Expand KV for GQA
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply sliding window mask if needed
        if self.sliding_window is not None and kv_seq_len > self.sliding_window:
            # Create sliding window mask
            sliding_mask = torch.ones_like(attn_weights, dtype=torch.bool)
            for i in range(seq_len):
                start_idx = max(0, kv_seq_len - seq_len + i - self.sliding_window)
                end_idx = kv_seq_len - seq_len + i + 1
                sliding_mask[:, :, i, start_idx:end_idx] = False
            attn_weights = attn_weights.masked_fill(sliding_mask, torch.finfo(attn_weights.dtype).min)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        
        # Prepare cache (store COMPRESSED representation)
        past_key_value = (compressed_kv,) if use_cache else None
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


class GptOssMlaMLP(nn.Module):
    """Single expert MLP (SwiGLU activation)."""
    
    def __init__(self, config: GptOssMlaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # SwiGLU: gate_proj and up_proj
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # SwiGLU: silu(gate) * up
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GptOssMlaMoE(nn.Module):
    """
    Mixture of Experts layer - preserves original GPT-OSS MoE structure.
    
    Uses top-k routing with load balancing.
    """
    
    def __init__(self, config: GptOssMlaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_aux_loss_coef = config.router_aux_loss_coef
        
        # Router
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            GptOssMlaMLP(config) for _ in range(self.num_experts)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)  # [batch * seq_len, hidden_size]
        
        # Router logits
        router_logits = self.router(hidden_states)  # [batch * seq_len, num_experts]
        
        # Top-k routing
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # Normalize weights
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        # Efficient batched expert computation
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == i).any(dim=-1)
            if not expert_mask.any():
                continue
            
            # Get tokens for this expert
            expert_input = hidden_states[expert_mask]
            expert_output = expert(expert_input)
            
            # Get weights for this expert
            expert_weights = torch.zeros(
                batch_size * seq_len,
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
            for k in range(self.num_experts_per_tok):
                expert_weights += routing_weights[:, k] * (selected_experts[:, k] == i).to(hidden_states.dtype)
            
            # Accumulate weighted output
            final_hidden_states[expert_mask] += expert_output * expert_weights[expert_mask, None]
        
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
        
        return final_hidden_states


class GptOssMlaDecoderLayer(nn.Module):
    """
    Single decoder layer with MLA attention + MoE.
    
    Structure:
    - Input LayerNorm
    - MLA Attention
    - Residual
    - Post-attention LayerNorm  
    - MoE FFN
    - Residual
    """
    
    def __init__(self, config: GptOssMlaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Attention
        self.self_attn = GptOssMlaAttention(config, layer_idx)
        self.input_layernorm = GptOssMlaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # MoE FFN
        # TODO: Implement MoE
        self.mlp = GptOssMlaMoE(config)
        self.post_attention_layernorm = GptOssMlaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            present_key_value: Optional cached KV
        """
        residual = hidden_states
        
        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs


class GptOssMlaPreTrainedModel(PreTrainedModel):
    """Base class for GPT-OSS MLA models."""
    config_class = GptOssMlaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GptOssMlaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class GptOssMlaModel(GptOssMlaPreTrainedModel):
    """
    Main model class without LM head.
    
    Transformer decoder consisting of config.num_hidden_layers layers.
    Each layer is a GptOssMlaDecoderLayer with MLA attention and MoE.
    """
    
    def __init__(self, config: GptOssMlaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            GptOssMlaDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final norm
        self.norm = GptOssMlaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPast:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            position_ids: [batch, seq_len]
            past_key_values: List of (compressed_kv,) tuples
            
        Returns:
            BaseModelOutputWithPast with:
                last_hidden_state: [batch, seq_len, hidden_size]
                past_key_values: List of (compressed_kv,) tuples
                hidden_states: Optional tuple of all layer outputs
                attentions: Optional tuple of attention weights
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Retrieve input embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        # Position IDs
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
        
        # Attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device
            )
        
        # Causal mask (lower triangular)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values
        )
        
        # Forward through layers
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)
        
        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values):
        """Create causal mask + combine with padding mask."""
        # Create causal mask
        batch_size, seq_length = input_shape
        combined_attention_mask = None
        device = inputs_embeds.device
        
        if seq_length > 1:
            # Causal mask
            combined_attention_mask = torch.triu(
                torch.ones(seq_length, seq_length, dtype=torch.bool, device=device),
                diagonal=1
            )
            # Expand to batch
            combined_attention_mask = combined_attention_mask[None, None, :, :].expand(
                batch_size, 1, seq_length, seq_length
            )
        
        # Handle past_key_values (extend mask for cached tokens)
        if past_key_values is not None:
            cache_length = past_key_values[0][0].shape[1]
            if combined_attention_mask is not None:
                # Extend mask for cached positions (all visible)
                past_mask = torch.zeros(
                    (batch_size, 1, seq_length, cache_length),
                    dtype=torch.bool,
                    device=device
                )
                combined_attention_mask = torch.cat([past_mask, combined_attention_mask], dim=-1)
        
        # Combine with padding mask
        if attention_mask is not None:
            # attention_mask: [batch, seq_len]
            expanded_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            if combined_attention_mask is not None:
                combined_attention_mask = combined_attention_mask | ~expanded_mask
            else:
                combined_attention_mask = ~expanded_mask
        
        # Convert to additive mask (0 = visible, -inf = masked)
        if combined_attention_mask is not None:
            combined_attention_mask = combined_attention_mask.to(inputs_embeds.dtype)
            combined_attention_mask = combined_attention_mask.masked_fill(
                combined_attention_mask.bool(), torch.finfo(inputs_embeds.dtype).min
            )
        
        return combined_attention_mask


class GptOssMlaForCausalLM(GptOssMlaPreTrainedModel, GenerationMixin):
    """GPT-OSS MLA model with language modeling head."""
    
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        self.model = GptOssMlaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def set_decoder(self, decoder):
        self.model = decoder
    
    def get_decoder(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        """
        Args:
            input_ids: [batch, seq_len]
            labels: [batch, seq_len] for computing language modeling loss
            
        Returns:
            CausalLMOutputWithPast with:
                loss: Optional language modeling loss
                logits: [batch, seq_len, vocab_size]
                past_key_values: List of cached KV states
                hidden_states: Optional tuple of hidden states
                attentions: Optional tuple of attention weights
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Decoder outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Prepare inputs for generation (autoregressive decoding)."""
        if past_key_values is not None:
            # Only use last token if we have past
            input_ids = input_ids[:, -1:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # Create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        
        # If `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorder cache for beam search."""
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


# Model classes will continue in next part...
