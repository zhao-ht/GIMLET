import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
import warnings
from transformers.utils import logging
from transformers.modeling_outputs import (
    BaseModelOutput,
)
from model.GIMLET.GIMLETEncoderStack import GraphormerT5EncoderStackCoAttention
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import copy
logger = logging.get_logger(__name__)

__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""






class GraphT5TransformerForGraphOnly(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config,graph_args):
        super().__init__(config)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.config.loss_reduction_method = getattr(graph_args,'loss_reduction_method')
        self.encoder = GraphormerT5EncoderStackCoAttention(encoder_config,graph_args, self.shared)
        self.classifier = nn.Linear(
            config.d_model, graph_args.num_classes, bias=True
        )
        self.readout=graph_args.graphonly_readout

    def forward(
        self,
        graph=None,
        input_ids = None,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ) :
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            # encoder_outputs = self.encoder(
            #     graph=graph,
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     inputs_embeds=inputs_embeds,
            #     head_mask=head_mask,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            # )
            if self.readout=='mean':
                encoder_outputs = self.encoder.forward_graph_only(graph=graph,
                # input_ids=None,
                # attention_mask=None,
                # encoder_hidden_states=None,
                # encoder_attention_mask=None,
                # inputs_embeds=None,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,)
            else:
                input_ids=self.config.eos_token_id*torch.ones([graph.x.shape[0],1]).long().to(graph.x.device)
                attention_mask=torch.ones([graph.x.shape[0],1]).long().to(graph.x.device)
                encoder_outputs = self.encoder(
                graph=graph,
                input_ids=input_ids,
                attention_mask=attention_mask,
                # inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        sequence_output = encoder_outputs[0] if isinstance(encoder_outputs,tuple) else encoder_outputs['last_hidden_state']


        if labels is not None:
            if self.encoder.graph_encoder.args.graphonly_problem_type is None:
                if self.encoder.graph_encoder.args.num_classes == 1:
                    self.encoder.graph_encoder.args.graphonly_problem_type = "regression"
                elif self.encoder.graph_encoder.args.num_classes > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.encoder.graph_encoder.args.graphonly_problem_type = "single_label_classification"
                else:
                    self.encoder.graph_encoder.args.graphonly_problem_type = "multi_label_classification"
        else:
            assert self.encoder.graph_encoder.args.graphonly_problem_type is not None

        if self.readout=='mean':
            if self.encoder.graph_encoder.args.graphonly_problem_type in ['regression','single_label_classification','multi_label_classification']:
                index=encoder_outputs['attention_mask_merged'].squeeze(2).squeeze(1).unsqueeze(2)==0
                sequence_output=(sequence_output*index).sum(1)/index.sum(1)
        else:
            sequence_output=sequence_output.squeeze(1)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.encoder.graph_encoder.args.graphonly_problem_type == "regression":
                loss_fct = MSELoss()
                if self.encoder.graph_encoder.args.num_classes == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.encoder.graph_encoder.args.graphonly_problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            elif self.encoder.graph_encoder.args.graphonly_problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )
        # hidden_states = encoder_outputs[0]
        #
        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)
        #
        # if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
        #     # get decoder inputs from shifting lm labels to the right
        #     decoder_input_ids = self._shift_right(labels)
        #
        # # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)
        #     hidden_states = hidden_states.to(self.decoder.first_device)
        #     if decoder_input_ids is not None:
        #         decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
        #     if attention_mask is not None:
        #         attention_mask = attention_mask.to(self.decoder.first_device)
        #     if decoder_attention_mask is not None:
        #         decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        #
        # # Decode
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     inputs_embeds=decoder_inputs_embeds,
        #     past_key_values=past_key_values,
        #     encoder_hidden_states=hidden_states,
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     cross_attn_head_mask=cross_attn_head_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        #
        # sequence_output = decoder_outputs[0]
        #
        # # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.encoder.first_device)
        #     self.lm_head = self.lm_head.to(self.encoder.first_device)
        #     sequence_output = sequence_output.to(self.lm_head.weight.device)
        #
        # if self.config.tie_word_embeddings:
        #     # Rescale output before projecting on vocab
        #     # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        #     sequence_output = sequence_output * (self.model_dim**-0.5)
        #
        # lm_logits = self.lm_head(sequence_output)
        #
        # loss = None
        # if labels is not None:
        #     if self.config.loss_reduction_method=='sentence':
        #         loss_fct = CrossEntropyLoss(ignore_index=-100,reduction='none')
        #         loss_perout = loss_fct(lm_logits.transpose(-1,-2), labels)
        #         cnt=(labels!=-100).sum(-1)
        #         cnt[cnt == 0] = 1 #avoid nan for reduction
        #         loss=(loss_perout.sum(-1) / cnt).mean()
        #         if loss>50:
        #             print(loss_perout)
        #             print(loss)
        #     elif self.config.loss_reduction_method=='token':
        #         loss_fct = CrossEntropyLoss(ignore_index=-100)
        #         loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        #     else:
        #         raise ValueError('Not supported loss reduction method yet')
        #     # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        #
        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output
        #
        # return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=lm_logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )








if __name__ == '__main__':
    model = T5ForConditionalGeneration.from_pretrained("molt5_base/")
    for p in model.named_parameters():
        if 'lm_head' in p[0] or 'shared' in p[0]:
            print(p[1])

    print(model.shared)
    print(model.lm_head)