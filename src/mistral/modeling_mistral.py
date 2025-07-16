import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import logging
from transformers.models.mistral.modeling_mistral import (
    MistralAttention, 
    MistralDecoderLayer,
    MistralRMSNorm,
    MistralRotaryEmbedding,
    MistralModel,
    MistralForCausalLM,
    MistralConfig,
    repeat_kv,
    apply_rotary_pos_emb,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.integrations import is_deepspeed_zero3_enabled
from typing import Optional, Tuple, Union, List

logger = logging.getLogger(__name__)

class MistralAttentionWithBeacon(MistralAttention):
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True


        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)

        self.beacon_q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.beacon_q_proj.weight.data.zero_()
        self.beacon_q_proj._is_hf_initialized = True
        self.beacon_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.beacon_k_proj.weight.data.zero_()
        self.beacon_k_proj._is_hf_initialized = True
        self.beacon_v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.beacon_v_proj.weight.data.zero_()
        self.beacon_v_proj._is_hf_initialized = True

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        is_beacon: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        flat_states = hidden_states.view(-1, hidden_states.shape[-1])
        if is_beacon is None:
            is_beacon = torch.zeros(bsz, q_len, dtype=torch.bool, device=hidden_states.device)
        flat_mask = is_beacon.reshape(-1)

        q_proj = self.q_proj(flat_states)
        q_proj[flat_mask] = self.beacon_q_proj(flat_states[flat_mask])

        k_proj = self.k_proj(flat_states)
        k_proj[flat_mask] = self.beacon_k_proj(flat_states[flat_mask])

        v_proj = self.v_proj(flat_states)
        v_proj[flat_mask] = self.beacon_v_proj(flat_states[flat_mask])

        query_states = q_proj.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = k_proj.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = v_proj.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    def _init_beacon_proj(self, missing_keys):
        """Initialize the beacon projection weight with that of the ordinal projection."""
        
        if is_deepspeed_zero3_enabled():
            raise ValueError("Does not support deepspeed zero 3!!!")
            # # FIXME: after deepspeed initialization, some weights becomes non-zero
            # # For Mistral, there are rows that are full of zeros
            # # For Mistral, there are values bigger than 1e29...

            # import deepspeed
            # params = [self.beacon_q_proj.weight, self.q_proj.weight]
            # if self.q_proj.bias is not None:
            #     params.extend([self.beacon_q_proj.bias, self.q_proj.bias])
            # with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            #     # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
            #     if (self.beacon_q_proj.weight.sum(-1) == 0).any() or (self.beacon_q_proj.weight > 1e29).any():
            #         self.beacon_q_proj.weight.data[:] = self.q_proj.weight.data
            #         if self.q_proj.bias is not None:
            #             self.beacon_q_proj.bias.data[:] = self.q_proj.bias.data
            # params = [self.beacon_k_proj.weight, self.k_proj.weight]
            # if self.k_proj.bias is not None:
            #     params.extend([self.beacon_k_proj.bias, self.k_proj.bias])
            # with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            #     # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
            #     if (self.beacon_k_proj.weight.sum(-1) == 0).any() or (self.beacon_k_proj.weight > 1e29).any():
            #         self.beacon_k_proj.weight.data[:] = self.k_proj.weight.data
            #         if self.k_proj.bias is not None:
            #             self.beacon_k_proj.bias.data[:] = self.k_proj.bias.data
            # params = [self.beacon_v_proj.weight, self.v_proj.weight]
            # if self.v_proj.bias is not None:
            #     params.extend([self.beacon_v_proj.bias, self.v_proj.bias])
            # with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            #     # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
            #     if (self.beacon_v_proj.weight.sum(-1) == 0).any() or (self.beacon_v_proj.weight > 1e29).any():
            #         self.beacon_v_proj.weight.data[:] = self.v_proj.weight.data
            #         if self.v_proj.bias is not None:
            #             self.beacon_v_proj.bias.data[:] = self.v_proj.bias.data
        else:
            # only copy the value in-place, without tieing the weight
            if any("beacon_q_proj" in missing_key for missing_key in missing_keys):
                # FIXME: some beacon weights are not initialized as zero for llama model, why? 
                # if (self.beacon_q_proj.weight == 0).all():
                self.beacon_q_proj.weight.data[:] = self.q_proj.weight.data
                if self.q_proj.bias is not None:
                    self.beacon_q_proj.bias.data[:] = self.q_proj.bias.data
                print(f"beacon_q_proj at layer {self.layer_idx} initialized!!!")
            if any("beacon_k_proj" in missing_key for missing_key in missing_keys):
                # if (self.beacon_k_proj.weight == 0).all():
                self.beacon_k_proj.weight.data[:] = self.k_proj.weight.data
                if self.k_proj.bias is not None:
                    self.beacon_k_proj.bias.data[:] = self.k_proj.bias.data
                print(f"beacon_k_proj at layer {self.layer_idx} initialized!!!")
            if any("beacon_v_proj" in missing_key for missing_key in missing_keys):
                # if (self.beacon_v_proj.weight == 0).all():
                self.beacon_v_proj.weight.data[:] = self.v_proj.weight.data
                if self.v_proj.bias is not None:
                    self.beacon_v_proj.bias.data[:] = self.v_proj.bias.data
                print(f"beacon_v_proj at layer {self.layer_idx} initialized!!!")


class MistralDecoderLayerWithBeacon(MistralDecoderLayer):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MistralAttentionWithBeacon(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        is_beacon: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            is_beacon=is_beacon,
            **kwargs,
        )
        hidden_states = residual + hidden_states

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


# ✅ 修改 MistralModel.forward：传入 is_beacon 给每一层 self_attn
class MistralModelWithBeacon(MistralModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # BEACON: add beacon embedding
        self.beacon_embed_tokens = nn.Embedding(1, config.hidden_size, self.padding_idx)
        self.beacon_embed_tokens._is_hf_initialized = True

        self.layers = nn.ModuleList(
            [MistralDecoderLayerWithBeacon(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        is_beacon: Optional[torch.BoolTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            # Default embed
            inputs_embeds = self.embed_tokens(input_ids)

            # Apply beacon embedding at specified positions
            if is_beacon is not None:
                inputs_embeds = inputs_embeds.clone()
                beacon_embed = self.beacon_embed_tokens.weight[0]  # shape: [hidden_size]
                inputs_embeds[is_beacon] = beacon_embed.to(inputs_embeds.dtype)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, use_cache, output_attentions
        )
        hidden_states = inputs_embeds


        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    is_beacon,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    is_beacon=is_beacon,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)


        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _init_beacon_embed(self, missing_keys):
        """Initialize the beacon token embedding with that of the eos token."""
        if is_deepspeed_zero3_enabled():
            raise ValueError("Does not support deepspeed zeros 3!!!")
            # import deepspeed
            # params = [self.beacon_embed_tokens.weight, self.embed_tokens.weight]
            # with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            #     # deepspeed will initialize the parameters to zero
            #     if (self.beacon_embed_tokens.weight == 0).all():
            #         if self.config.beacon_embed_init == "bos":
            #             self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.bos_token_id]
            #         elif self.config.beacon_embed_init == "eos":
            #             if isinstance(self.config.eos_token_id, list):
            #                 eos_token_id = self.config.eos_token_id[0]
            #             else:
            #                 eos_token_id = self.config.eos_token_id
            #             self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[eos_token_id]
            #         else:
            #             raise NotImplementedError(f"Make sure beacon_embed_init is either eos or bos, found {self.config.beacon_embed_init}")
        else:
            if any("beacon_embed_tokens" in missing_key for missing_key in missing_keys):
                # if self.config.beacon_embed_init == "bos":
                #     self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.bos_token_id]
                # elif self.config.beacon_embed_init == "eos":
                if isinstance(self.config.eos_token_id, list):
                    eos_token_id = self.config.eos_token_id[0]
                else:
                    eos_token_id = self.config.eos_token_id
                self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[eos_token_id]
                print("beacon_embed_tokens initialized!!!")
                # else:
                #     raise NotImplementedError(f"Make sure beacon_embed_init is either eos or bos, found {self.config.beacon_embed_init}")

class MistralForCausalLMWithBeacon(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModelWithBeacon(config)
        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        is_beacon: Optional[torch.BoolTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            cache_position=cache_position,
            is_beacon=is_beacon,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]) if num_logits_to_keep > 0 else self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
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
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override the default from_pretrained."""
        kwargs.update(output_loading_info=True)
        model, loading_info = super().from_pretrained(*args, **kwargs)

        # NOTE: set memory after from_pretrained because there may be another transformer model inside the Memory object, which may cause weird erros during loading
        # config = model.config
        # model.memory = Memory(
        #     model_config=config,
        #     k_seq_dim=2,
        #     v_seq_dim=2,
        # )

        missing_keys = loading_info["missing_keys"]
        # NOTE: the beacon parameters may or may not be loaded from the checkpoint
        # if it is loaded from the checkpoint, we should not re-initilize it
        model.model._init_beacon_embed(missing_keys)
        # initialize weights of possible q,k,v,o,mlp
        for layer in model.model.layers:
            layer.self_attn._init_beacon_proj(missing_keys)

        return model
    
    # def generate(self, )