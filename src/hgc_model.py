import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Model
from .hgc_layer import HGC_Layer

class GPT2_With_HGC(GPT2LMHeadModel):
    """
    A custom GPT-2 model that integrates the HGC Layer to condition its output.
    """
    def __init__(self, config):
        super().__init__(config)
        # Instantiate our HGC Layer
        self.hgc_layer = HGC_Layer(hidden_size=config.n_embd, hkm_dim=2048)

        # We need a linear layer to project the HGC output back to the model's hidden size
        self.hgc_output_projection = nn.Linear(self.hgc_layer.hkm_dim, config.n_embd)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 1. Get the standard transformer outputs
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # 2. Pass the final hidden states through our HGC Layer
        hgc_info = self.hgc_layer(hidden_states)

        # 3. Project the HGC information back to the model's hidden size
        projected_hgc_info = self.hgc_output_projection(hgc_info)

        # 4. Condition the model's output by adding the HGC info
        conditioned_hidden_states = hidden_states + projected_hgc_info

        # 5. Pass the conditioned states to the final language model head
        lm_logits = self.lm_head(conditioned_hidden_states)

        # --- Standard loss calculation from Hugging Face ---
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Create and return the model output object
        from transformers.modeling_outputs import CausalLMOutputWithPast
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
