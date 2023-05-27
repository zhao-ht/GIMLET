import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import T5Model, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.t5.modeling_t5 import T5Stack
from model.molecule_gnn_model import GNN
import warnings
from torch.nn import CrossEntropyLoss
from transformers.utils import logging
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
import copy
from .graphormer.models.graphormer import GraphormerModel
from .graphormer.modules.graphormer_graph_encoder import GraphEncoder
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

logger = logging.get_logger(__name__)



class GraphormerT5EncoderStackSequential(T5Stack):
    def __init__(self, config, graph_args,embed_tokens=None):
        super().__init__(config,embed_tokens=embed_tokens)
        # self.graph_encoder = GNN(
        #     num_layer=5,
        #     emb_dim=300,
        #     gnn_type='gin',
        #     drop_ratio=0.0,
        #     JK='last',
        # )

        self.graph_encoder = GraphormerModel.build_model(graph_args)
        self.graph_projector = nn.Sequential(
            nn.Linear(graph_args.encoder_embed_dim, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size, config.hidden_size)
        )


    def forward(
        self,
        graph=None,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)

        if inputs_embeds is None:
            input_embeds_new = self.embed_tokens(input_ids)
        else:
            input_embeds_new = inputs_embeds
        device = input_ids.device
        B, _ = input_ids.shape

        # if self.has_graph:

        graph_rep = self.graph_encoder(graph)[-1][0,:,:] #cls representation of last layer
        # graph_rep = self.graph_encoder.pool(graph_rep, graph['batch'])

        graph_rep = self.graph_projector(graph_rep)
        # graph_rep = torch.nn.functional.normalize(graph_rep, dim=1)
        input_embeds_new = torch.cat([graph_rep.unsqueeze(1), input_embeds_new[:, :-1, :]], dim=1)
        attention_mask = torch.cat([torch.ones(B, 1).to(device), attention_mask[:, :-1]], dim=1)

        input_ids=None
        inputs_embeds=input_embeds_new


        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class GraphormerT5EncoderStackCoAttention(T5Stack):
    def __init__(self, config, graph_args,embed_tokens=None):
        super().__init__(config,embed_tokens=embed_tokens)
        # self.graph_encoder = GNN(
        #     num_layer=5,
        #     emb_dim=300,
        #     gnn_type='gin',
        #     drop_ratio=0.0,
        #     JK='last',
        # )
        if graph_args.unimodel:
            if graph_args.encoder_embed_dim!=self.config.d_model:
                print('Warning! Inconsistent graph encoder_embed_dim and transformer d_model for Unimodel')
                graph_args.encoder_embed_dim=self.config.d_model
            if graph_args.encoder_attention_heads!=self.config.num_heads:
                print('Warning! Inconsistent graph encoder_attention_heads and transformer num_heads for Unimodel')
                graph_args.encoder_attention_heads=self.config.num_heads
        if graph_args.unimodel:
            self.graph_encoder = GraphEncoder(
            num_atoms=graph_args.num_atoms,
            num_in_degree=graph_args.num_in_degree,
            num_out_degree=graph_args.num_out_degree,
            num_edges=graph_args.num_edges,
            num_spatial=graph_args.num_spatial,
            num_edge_dis=graph_args.num_edge_dis,
            edge_type=graph_args.edge_type,
            multi_hop_max_dist=graph_args.multi_hop_max_dist,
            # >
            num_encoder_layers=graph_args.encoder_layers,
            embedding_dim=graph_args.encoder_embed_dim,
            num_attention_heads=graph_args.encoder_attention_heads,
            dropout=graph_args.dropout,
            encoder_normalize_before=graph_args.encoder_normalize_before,
            apply_graphormer_init=graph_args.apply_graphormer_init,

        )
            self.graph_encoder.args=graph_args
        else:
            self.graph_encoder = GraphormerModel.build_model(graph_args)
            self.graph_projector = nn.Sequential(
                nn.Linear(graph_args.encoder_embed_dim, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, config.hidden_size)
            )

        self.position_embedding_graph=nn.Embedding(1, self.block[1].layer[0].SelfAttention.n_heads)


    def forward(
        self,
        graph=None,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        graph=graph.to(input_ids.device)
        attention_mask_graph=graph['attn_bias'][:,0,:]

        if self.graph_encoder.args.unimodel:
            hidden_state_graph, attn_bias_graph, padding_mask_graph = self.graph_encoder(graph)
        else:
            hidden_state_graph,attn_bias_graph,padding_mask_graph=self.graph_encoder.encoder.graph_encoder.embedding_graph(graph)
        attention_bias_graph_unimodel = attn_bias_graph.masked_fill(
            padding_mask_graph.unsqueeze(1).unsqueeze(2).to(torch.bool),
            float("-inf"))


        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)


        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        # construct position_bias for T5, where graph tokens are all treated as a special distance to each text token
        position_bias=self.block[0].layer[0].SelfAttention.compute_bias(hidden_states.shape[1], hidden_states.shape[1], device=hidden_states.device)
        position_embedding_graph_spaned=self.position_embedding_graph(
            torch.zeros(1,hidden_states.shape[1],hidden_state_graph.shape[0]).long().to(self.position_embedding_graph.weight.device))\
            .transpose(3,1)
        position_embedding_graph_spaned_2=self.position_embedding_graph(
            torch.zeros(1,hidden_state_graph.shape[0],hidden_states.shape[1]+hidden_state_graph.shape[0]).long().to(self.position_embedding_graph.weight.device))\
            .transpose(3,1)
        position_bias_merged=torch.cat([position_embedding_graph_spaned,position_bias],2)
        position_bias_merged = torch.cat([position_embedding_graph_spaned_2, position_bias_merged], 3)

        if self.graph_encoder.args.unimodel:
            position_bias_merged = position_bias_merged.repeat(attention_bias_graph_unimodel.shape[0], 1, 1, 1) #attention_bias_graph_unimodel.shape[0] is batch size
            position_bias_merged[:, :, 0:attention_bias_graph_unimodel.shape[2],
                                0:attention_bias_graph_unimodel.shape[3]] = attention_bias_graph_unimodel
            if self.graph_encoder.args.maskt2g:
                position_bias_merged[:,:,0:attention_bias_graph_unimodel.shape[2],attention_bias_graph_unimodel.shape[3]:]=float("-inf")

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if self.graph_encoder.args.unimodel:
                #Because hidden_state_graph size is length,batch,embedding_dim, need transpose before input to T5
                # transpose to be consistant with the non-unimodel
                # hidden_state_graph=hidden_state_graph.transpose(0,1)

                layer_head_mask = head_mask[i]
                cross_attn_layer_head_mask = cross_attn_head_mask[i]
                # Model parallel
                if self.model_parallel:
                    torch.cuda.set_device(hidden_states.device)
                    # Ensure that attention_mask is always on the same device as hidden_states
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(hidden_states.device)
                    if position_bias is not None:
                        position_bias = position_bias.to(hidden_states.device)
                    if encoder_hidden_states is not None:
                        encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                    if encoder_extended_attention_mask is not None:
                        encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                    if encoder_decoder_position_bias is not None:
                        encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                    if layer_head_mask is not None:
                        layer_head_mask = layer_head_mask.to(hidden_states.device)
                    if cross_attn_layer_head_mask is not None:
                        cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    if use_cache:
                        logger.warning(
                            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                        )
                        use_cache = False

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return tuple(module(*inputs, use_cache, output_attentions))

                        return custom_forward

                    layer_outputs = checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        extended_attention_mask,
                        position_bias_merged,
                        encoder_hidden_states,
                        encoder_extended_attention_mask,
                        encoder_decoder_position_bias,
                        layer_head_mask,
                        cross_attn_layer_head_mask,
                        None,  # past_key_value is always None with gradient checkpointing
                    )
                else:
                    # transpose [length,batch,dim] to [batch,length,dim]
                    # if not self.graph_encoder.args.unimodel:
                    #     hidden_states_graph_mapped = (self.graph_projector(hidden_state_graph)).transpose(0, 1)
                    # else:
                    hidden_states_graph_mapped = hidden_state_graph.transpose(0, 1)

                    hidden_states_merged = torch.cat([hidden_states_graph_mapped, hidden_states], 1)
                    attention_mask_merged = torch.cat(
                        [attention_mask_graph.unsqueeze(1).unsqueeze(1), extended_attention_mask], 3)



                    layer_outputs = layer_module(
                        hidden_states_merged,
                        # attention_mask=attention_mask_merged,
                        position_bias=position_bias_merged + attention_mask_merged,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_extended_attention_mask,
                        encoder_decoder_position_bias=encoder_decoder_position_bias,
                        layer_head_mask=layer_head_mask,
                        cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                        past_key_value=past_key_value,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )

                # layer_outputs is a tuple with:
                # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
                if use_cache is False:
                    layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

                hidden_states_merged, present_key_value_state = layer_outputs[:2]

                hidden_states = hidden_states_merged[:, hidden_states_graph_mapped.shape[1]:, :]
                # transpose again for consistance
                hidden_state_graph=(hidden_states_merged[:, 0:hidden_states_graph_mapped.shape[1], :]).transpose(0, 1)

                assert hidden_states.shape[1] == extended_attention_mask.shape[3]

                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
                # (cross-attention position bias), (cross-attention weights)
                position_bias = layer_outputs[2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
                # append next layer key value states
                if use_cache:
                    present_key_value_states = present_key_value_states + (present_key_value_state,)

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[3],)
                    if self.is_decoder:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

                # Model Parallel: If it's the last layer for that device, put things on the next device
                if self.model_parallel:
                    for k, v in self.device_map.items():
                        if i == v[-1] and "cuda:" + str(k) != self.last_device:
                            hidden_states = hidden_states.to("cuda:" + str(k + 1))

            else:
                hidden_state_graph, _ = self.graph_encoder.encoder.graph_encoder.layers[i](
                    hidden_state_graph,
                    self_attn_padding_mask=padding_mask_graph,
                    self_attn_mask=None,
                    self_attn_bias=attn_bias_graph,
                )

                layer_head_mask = head_mask[i]
                cross_attn_layer_head_mask = cross_attn_head_mask[i]
                # Model parallel
                if self.model_parallel:
                    torch.cuda.set_device(hidden_states.device)
                    # Ensure that attention_mask is always on the same device as hidden_states
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(hidden_states.device)
                    if position_bias is not None:
                        position_bias = position_bias.to(hidden_states.device)
                    if encoder_hidden_states is not None:
                        encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                    if encoder_extended_attention_mask is not None:
                        encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                    if encoder_decoder_position_bias is not None:
                        encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                    if layer_head_mask is not None:
                        layer_head_mask = layer_head_mask.to(hidden_states.device)
                    if cross_attn_layer_head_mask is not None:
                        cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    if use_cache:
                        logger.warning(
                            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                        )
                        use_cache = False

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return tuple(module(*inputs, use_cache, output_attentions))

                        return custom_forward

                    layer_outputs = checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        extended_attention_mask,
                        position_bias_merged,
                        encoder_hidden_states,
                        encoder_extended_attention_mask,
                        encoder_decoder_position_bias,
                        layer_head_mask,
                        cross_attn_layer_head_mask,
                        None,  # past_key_value is always None with gradient checkpointing
                    )
                else:
                    #transpose [length,batch,dim] to [batch,length,dim]

                    hidden_states_graph_mapped=(self.graph_projector(hidden_state_graph)).transpose(0,1)

                    hidden_states_merged=torch.cat([hidden_states_graph_mapped,hidden_states],1)
                    attention_mask_merged=torch.cat([attention_mask_graph.unsqueeze(1).unsqueeze(1),extended_attention_mask],3)


                    layer_outputs = layer_module(
                        hidden_states_merged,
                        # attention_mask=attention_mask_merged,
                        position_bias=position_bias_merged+attention_mask_merged,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_extended_attention_mask,
                        encoder_decoder_position_bias=encoder_decoder_position_bias,
                        layer_head_mask=layer_head_mask,
                        cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                        past_key_value=past_key_value,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )

                # layer_outputs is a tuple with:
                # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
                if use_cache is False:
                    layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

                hidden_states_merged, present_key_value_state = layer_outputs[:2]

                hidden_states=hidden_states_merged[:,hidden_states_graph_mapped.shape[1]:,:]
                assert hidden_states.shape[1]==extended_attention_mask.shape[3]



                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
                # (cross-attention position bias), (cross-attention weights)
                position_bias = layer_outputs[2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
                # append next layer key value states
                if use_cache:
                    present_key_value_states = present_key_value_states + (present_key_value_state,)

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[3],)
                    if self.is_decoder:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

                # Model Parallel: If it's the last layer for that device, put things on the next device
                if self.model_parallel:
                    for k, v in self.device_map.items():
                        if i == v[-1] and "cuda:" + str(k) != self.last_device:
                            hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)

        #for debug
        # hidden_states = self.final_layer_norm(hidden_state_graph.transpose(0,1))

        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    def forward_graph_only(
        self,
        graph=None,
        # input_ids=None,
        # attention_mask=None,
        # encoder_hidden_states=None,
        # encoder_attention_mask=None,
        # inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        graph=graph.to(self.device)
        attention_mask_graph=graph['attn_bias'][:,0,:]

        if self.graph_encoder.args.unimodel:
            hidden_state_graph, attn_bias_graph, padding_mask_graph = self.graph_encoder(graph)
        else:
            hidden_state_graph,attn_bias_graph,padding_mask_graph=self.graph_encoder.encoder.graph_encoder.embedding_graph(graph)
        attention_bias_graph_unimodel = attn_bias_graph.masked_fill(
            padding_mask_graph.unsqueeze(1).unsqueeze(2).to(torch.bool),
            float("-inf"))


        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)



        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if input_ids is not None and inputs_embeds is not None:
        #     err_msg_prefix = "decoder_" if self.is_decoder else ""
        #     raise ValueError(
        #         f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
        #     )
        # elif input_ids is not None:
        #     input_shape = input_ids.size()
        #     input_ids = input_ids.view(-1, input_shape[-1])
        # elif inputs_embeds is not None:
        #     input_shape = inputs_embeds.size()[:-1]
        # else:
        #     err_msg_prefix = "decoder_" if self.is_decoder else ""
        #     raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        # if inputs_embeds is None:
        #     assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
        #     inputs_embeds = self.embed_tokens(input_ids)
        #
        # batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        # mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        # if attention_mask is None:
        #     attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        # if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
        #     encoder_seq_length = encoder_hidden_states.shape[1]
        #     encoder_attention_mask = torch.ones(
        #         batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
        #     )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # if self.is_decoder and encoder_hidden_states is not None:
        #     encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        #     encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        #     if encoder_attention_mask is None:
        #         encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
        #     encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        # else:
        #     encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        # hidden_states = self.dropout(inputs_embeds)

        # construct position_bias for T5, where graph tokens are all treated as a special distance to each text token
        # position_bias=self.block[0].layer[0].SelfAttention.compute_bias(hidden_states.shape[1], hidden_states.shape[1], device=hidden_states.device)
        # position_embedding_graph_spaned=self.position_embedding_graph(
        #     torch.zeros(1,hidden_states.shape[1],hidden_state_graph.shape[0]).long().to(self.position_embedding_graph.weight.device))\
        #     .transpose(3,1)
        # position_embedding_graph_spaned_2=self.position_embedding_graph(
        #     torch.zeros(1,hidden_state_graph.shape[0],hidden_states.shape[1]+hidden_state_graph.shape[0]).long().to(self.position_embedding_graph.weight.device))\
        #     .transpose(3,1)
        # position_bias_merged=torch.cat([position_embedding_graph_spaned,position_bias],2)
        # position_bias_merged = torch.cat([position_embedding_graph_spaned_2, position_bias_merged], 3)

        if self.graph_encoder.args.unimodel:
            position_bias_merged=attention_bias_graph_unimodel
            # position_bias_merged = position_bias_merged.repeat(attention_bias_graph_unimodel.shape[0], 1, 1, 1) #attention_bias_graph_unimodel.shape[0] is batch size
            # position_bias_merged[:, :, 0:attention_bias_graph_unimodel.shape[2],
            #                     0:attention_bias_graph_unimodel.shape[3]] = attention_bias_graph_unimodel
            # if self.graph_encoder.args.maskt2g:
            #     position_bias_merged[:,:,0:attention_bias_graph_unimodel.shape[2],attention_bias_graph_unimodel.shape[3]:]=float("-inf")

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            assert self.graph_encoder.args.unimodel
            #Because hidden_state_graph size is length,batch,embedding_dim, need transpose before input to T5
            # transpose to be consistant with the non-unimodel
            # hidden_state_graph=hidden_state_graph.transpose(0,1)

            # layer_head_mask = head_mask[i]
            # cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                # if encoder_hidden_states is not None:
                #     encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                # if encoder_extended_attention_mask is not None:
                #     encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                # if layer_head_mask is not None:
                #     layer_head_mask = layer_head_mask.to(hidden_states.device)
                # if cross_attn_layer_head_mask is not None:
                #     cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias_merged,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                # transpose [length,batch,dim] to [batch,length,dim]
                # if not self.graph_encoder.args.unimodel:
                #     hidden_states_graph_mapped = (self.graph_projector(hidden_state_graph)).transpose(0, 1)
                # else:
                hidden_states_graph_mapped = hidden_state_graph.transpose(0, 1)

                hidden_states_merged = hidden_states_graph_mapped
                attention_mask_merged = attention_mask_graph.unsqueeze(1).unsqueeze(1)

                layer_outputs = layer_module(
                    hidden_states_merged,
                    # attention_mask=attention_mask_merged,
                    position_bias=position_bias_merged + attention_mask_merged,
                    # encoder_hidden_states=encoder_hidden_states,
                    # encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    # layer_head_mask=layer_head_mask,
                    # cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states_merged, present_key_value_state = layer_outputs[:2]

            hidden_states = hidden_states_merged
            # transpose again for consistance
            hidden_state_graph=hidden_states_merged.transpose(0, 1)

            # assert hidden_states.shape[1] == extended_attention_mask.shape[3]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            # if self.is_decoder and encoder_hidden_states is not None:
            #     encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))



        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    attention_mask_merged
            )
        return dict(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            attention_mask_merged=attention_mask_merged
        )


class GinT5EncoderStack(T5Stack):
    def __init__(self, config,graph_args, embed_tokens=None):
        super().__init__(config)
        # self.graph_encoder = GNN(
        #     num_layer=5,
        #     emb_dim=300,
        #     gnn_type='gin',
        #     drop_ratio=0.0,
        #     JK='last',
        # )
        # The Gin in this model only use base feature, i.e. the first two rows of x, and the first two rows of edge_feat, even if rich feature is given

        self.graph_encoder = GNN(**vars(graph_args))
        self.graph_projector = nn.Sequential(
            nn.Linear(graph_args.emb_dim, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size, config.hidden_size)
        )


    def forward(
        self,
        graph=None,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)

        if inputs_embeds is None:
            input_embeds_new = self.embed_tokens(input_ids)
        else:
            input_embeds_new = inputs_embeds
        device = input_ids.device
        B, _ = input_ids.shape

        # if self.has_graph:

        graph_rep = self.graph_encoder(graph)
        graph_rep = self.graph_encoder.pool(graph_rep, graph['batch'])

        graph_rep = self.graph_projector(graph_rep)
        # graph_rep = torch.nn.functional.normalize(graph_rep, dim=1)
        input_embeds_new = torch.cat([graph_rep.unsqueeze(1), input_embeds_new[:, :-1, :]], dim=1)
        attention_mask = torch.cat([torch.ones(B, 1).to(device), attention_mask[:, :-1]], dim=1)

        input_ids=None
        inputs_embeds=input_embeds_new


        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

# class GraphormerT5EncoderStackCoAttentionGraphOnly(GraphormerT5EncoderStackCoAttention):
#     def __init__(self, config, graph_args,embed_tokens=None):
#         super().__init__(config,graph_args,embed_tokens=embed_tokens)
#
#
#
#     def forward(
#         self,
#         graph=None,
#         # input_ids=None,
#         # attention_mask=None,
#         # encoder_hidden_states=None,
#         # encoder_attention_mask=None,
#         # inputs_embeds=None,
#         head_mask=None,
#         cross_attn_head_mask=None,
#         past_key_values=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         graph=graph.to(self.device)
#         attention_mask_graph=graph['attn_bias'][:,0,:]
#
#         if self.graph_encoder.args.unimodel:
#             hidden_state_graph, attn_bias_graph, padding_mask_graph = self.graph_encoder(graph)
#         else:
#             hidden_state_graph,attn_bias_graph,padding_mask_graph=self.graph_encoder.encoder.graph_encoder.embedding_graph(graph)
#         attention_bias_graph_unimodel = attn_bias_graph.masked_fill(
#             padding_mask_graph.unsqueeze(1).unsqueeze(2).to(torch.bool),
#             float("-inf"))
#
#
#         # Model parallel
#         if self.model_parallel:
#             torch.cuda.set_device(self.first_device)
#             self.embed_tokens = self.embed_tokens.to(self.first_device)
#
#
#
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         # if input_ids is not None and inputs_embeds is not None:
#         #     err_msg_prefix = "decoder_" if self.is_decoder else ""
#         #     raise ValueError(
#         #         f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
#         #     )
#         # elif input_ids is not None:
#         #     input_shape = input_ids.size()
#         #     input_ids = input_ids.view(-1, input_shape[-1])
#         # elif inputs_embeds is not None:
#         #     input_shape = inputs_embeds.size()[:-1]
#         # else:
#         #     err_msg_prefix = "decoder_" if self.is_decoder else ""
#         #     raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")
#
#         # if inputs_embeds is None:
#         #     assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
#         #     inputs_embeds = self.embed_tokens(input_ids)
#         #
#         # batch_size, seq_length = input_shape
#
#         # required mask seq length can be calculated via length of past
#         # mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
#
#         if use_cache is True:
#             assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"
#
#         # if attention_mask is None:
#         #     attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
#         # if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
#         #     encoder_seq_length = encoder_hidden_states.shape[1]
#         #     encoder_attention_mask = torch.ones(
#         #         batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
#         #     )
#
#         # initialize past_key_values with `None` if past does not exist
#         if past_key_values is None:
#             past_key_values = [None] * len(self.block)
#
#         # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
#         # ourselves in which case we just need to make it broadcastable to all heads.
#         # extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
#
#         # If a 2D or 3D attention mask is provided for the cross-attention
#         # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
#         # if self.is_decoder and encoder_hidden_states is not None:
#         #     encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
#         #     encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
#         #     if encoder_attention_mask is None:
#         #         encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
#         #     encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
#         # else:
#         #     encoder_extended_attention_mask = None
#
#         # Prepare head mask if needed
#         head_mask = self.get_head_mask(head_mask, self.config.num_layers)
#         cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
#         present_key_value_states = () if use_cache else None
#         all_hidden_states = () if output_hidden_states else None
#         all_attentions = () if output_attentions else None
#         all_cross_attentions = () if (output_attentions and self.is_decoder) else None
#         position_bias = None
#         encoder_decoder_position_bias = None
#
#         # hidden_states = self.dropout(inputs_embeds)
#
#         # construct position_bias for T5, where graph tokens are all treated as a special distance to each text token
#         # position_bias=self.block[0].layer[0].SelfAttention.compute_bias(hidden_states.shape[1], hidden_states.shape[1], device=hidden_states.device)
#         # position_embedding_graph_spaned=self.position_embedding_graph(
#         #     torch.zeros(1,hidden_states.shape[1],hidden_state_graph.shape[0]).long().to(self.position_embedding_graph.weight.device))\
#         #     .transpose(3,1)
#         # position_embedding_graph_spaned_2=self.position_embedding_graph(
#         #     torch.zeros(1,hidden_state_graph.shape[0],hidden_states.shape[1]+hidden_state_graph.shape[0]).long().to(self.position_embedding_graph.weight.device))\
#         #     .transpose(3,1)
#         # position_bias_merged=torch.cat([position_embedding_graph_spaned,position_bias],2)
#         # position_bias_merged = torch.cat([position_embedding_graph_spaned_2, position_bias_merged], 3)
#
#         if self.graph_encoder.args.unimodel:
#             position_bias_merged=attention_bias_graph_unimodel
#             # position_bias_merged = position_bias_merged.repeat(attention_bias_graph_unimodel.shape[0], 1, 1, 1) #attention_bias_graph_unimodel.shape[0] is batch size
#             # position_bias_merged[:, :, 0:attention_bias_graph_unimodel.shape[2],
#             #                     0:attention_bias_graph_unimodel.shape[3]] = attention_bias_graph_unimodel
#             # if self.graph_encoder.args.maskt2g:
#             #     position_bias_merged[:,:,0:attention_bias_graph_unimodel.shape[2],attention_bias_graph_unimodel.shape[3]:]=float("-inf")
#
#         for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
#             assert self.graph_encoder.args.unimodel
#             #Because hidden_state_graph size is length,batch,embedding_dim, need transpose before input to T5
#             # transpose to be consistant with the non-unimodel
#             # hidden_state_graph=hidden_state_graph.transpose(0,1)
#
#             # layer_head_mask = head_mask[i]
#             # cross_attn_layer_head_mask = cross_attn_head_mask[i]
#             # Model parallel
#             if self.model_parallel:
#                 torch.cuda.set_device(hidden_states.device)
#                 # Ensure that attention_mask is always on the same device as hidden_states
#                 if attention_mask is not None:
#                     attention_mask = attention_mask.to(hidden_states.device)
#                 if position_bias is not None:
#                     position_bias = position_bias.to(hidden_states.device)
#                 # if encoder_hidden_states is not None:
#                 #     encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
#                 # if encoder_extended_attention_mask is not None:
#                 #     encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
#                 if encoder_decoder_position_bias is not None:
#                     encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
#                 # if layer_head_mask is not None:
#                 #     layer_head_mask = layer_head_mask.to(hidden_states.device)
#                 # if cross_attn_layer_head_mask is not None:
#                 #     cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)
#
#             if self.gradient_checkpointing and self.training:
#                 if use_cache:
#                     logger.warning(
#                         "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
#                     )
#                     use_cache = False
#
#                 def create_custom_forward(module):
#                     def custom_forward(*inputs):
#                         return tuple(module(*inputs, use_cache, output_attentions))
#
#                     return custom_forward
#
#                 layer_outputs = checkpoint(
#                     create_custom_forward(layer_module),
#                     hidden_states,
#                     extended_attention_mask,
#                     position_bias_merged,
#                     encoder_hidden_states,
#                     encoder_extended_attention_mask,
#                     encoder_decoder_position_bias,
#                     layer_head_mask,
#                     cross_attn_layer_head_mask,
#                     None,  # past_key_value is always None with gradient checkpointing
#                 )
#             else:
#                 # transpose [length,batch,dim] to [batch,length,dim]
#                 # if not self.graph_encoder.args.unimodel:
#                 #     hidden_states_graph_mapped = (self.graph_projector(hidden_state_graph)).transpose(0, 1)
#                 # else:
#                 hidden_states_graph_mapped = hidden_state_graph.transpose(0, 1)
#
#                 hidden_states_merged = hidden_states_graph_mapped
#                 attention_mask_merged = attention_mask_graph.unsqueeze(1).unsqueeze(1)
#
#                 layer_outputs = layer_module(
#                     hidden_states_merged,
#                     # attention_mask=attention_mask_merged,
#                     position_bias=position_bias_merged + attention_mask_merged,
#                     # encoder_hidden_states=encoder_hidden_states,
#                     # encoder_attention_mask=encoder_extended_attention_mask,
#                     encoder_decoder_position_bias=encoder_decoder_position_bias,
#                     # layer_head_mask=layer_head_mask,
#                     # cross_attn_layer_head_mask=cross_attn_layer_head_mask,
#                     past_key_value=past_key_value,
#                     use_cache=use_cache,
#                     output_attentions=output_attentions,
#                 )
#
#             # layer_outputs is a tuple with:
#             # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
#             if use_cache is False:
#                 layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
#
#             hidden_states_merged, present_key_value_state = layer_outputs[:2]
#
#             hidden_states = hidden_states_merged
#             # transpose again for consistance
#             hidden_state_graph=hidden_states_merged.transpose(0, 1)
#
#             # assert hidden_states.shape[1] == extended_attention_mask.shape[3]
#
#             # We share the position biases between the layers - the first layer store them
#             # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
#             # (cross-attention position bias), (cross-attention weights)
#             position_bias = layer_outputs[2]
#             # if self.is_decoder and encoder_hidden_states is not None:
#             #     encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
#             # append next layer key value states
#             if use_cache:
#                 present_key_value_states = present_key_value_states + (present_key_value_state,)
#
#             if output_attentions:
#                 all_attentions = all_attentions + (layer_outputs[3],)
#                 if self.is_decoder:
#                     all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
#
#             # Model Parallel: If it's the last layer for that device, put things on the next device
#             if self.model_parallel:
#                 for k, v in self.device_map.items():
#                     if i == v[-1] and "cuda:" + str(k) != self.last_device:
#                         hidden_states = hidden_states.to("cuda:" + str(k + 1))
#
#
#
#         hidden_states = self.final_layer_norm(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#
#         # Add last layer
#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)
#
#         if not return_dict:
#             return tuple(
#                 v
#                 for v in [
#                     hidden_states,
#                     present_key_value_states,
#                     all_hidden_states,
#                     all_attentions,
#                     all_cross_attentions,
#                 ]
#                 if v is not None
#             )
#         return BaseModelOutputWithPastAndCrossAttentions(
#             last_hidden_state=hidden_states,
#             past_key_values=present_key_value_states,
#             hidden_states=all_hidden_states,
#             attentions=all_attentions,
#             cross_attentions=all_cross_attentions,
#         )



GraphT5EncoderStack_dict=GraphTransformer_dict={'gin':{'sequential':GinT5EncoderStack},
                       'graphormer':{'sequential':GraphormerT5EncoderStackSequential,
                                           'coattention':GraphormerT5EncoderStackCoAttention}}



