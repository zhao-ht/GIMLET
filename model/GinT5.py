import torch
import torch.nn as nn
from transformers import T5Model, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from model.gin_model import GNN


class GinDecoder(nn.Module):
    def __init__(self, has_graph=False, MoMuK=True, model_size='base'):
        super(GinDecoder, self).__init__()
        self.has_graph = has_graph
        self.main_model = T5ForConditionalGeneration.from_pretrained("laituan245/molt5-"+model_size)
        print(self.main_model.config.hidden_size)

        # for p in self.main_model.named_parameters():
        #     p[1].requires_grad = False

        if has_graph:
            self.graph_encoder = GNN(
                num_layer=5,
                emb_dim=300,
                gnn_type='gin',
                drop_ratio=0.0,
                JK='last',
            )
            
            if MoMuK:
                ckpt = torch.load("./checkpoints/littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt")
            else:
                ckpt = torch.load("./checkpoints/littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt")
            ckpt = ckpt['state_dict']
            pretrained_dict = {k[14:]: v for k, v in ckpt.items()}
            missing_keys, unexpected_keys = self.graph_encoder.load_state_dict(pretrained_dict, strict=False)

            for p in self.graph_encoder.named_parameters():
                p[1].requires_grad = False

            self.graph_projector = nn.Sequential(
                nn.Linear(300, self.main_model.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.main_model.config.hidden_size, self.main_model.config.hidden_size)
            )
            # self.graph_projector = nn.Linear(300, self.main_model.config.hidden_size)



    def forward(self, batch, input_ids, encoder_attention_mask, decoder_attention_mask, label):
        # typ = torch.zeros(input_ids.shape).long().to(device)


        # print("GIN output shape: ", node_reps.shape)
        # print("T5 decoder input token shape: ", input_ids.shape)  # B,L
        # print("T5 decoder input mask shape: ", tgt_mask.shape)  # B,L
        # print("GIN node mask: ", encoder_attention_mask[0])

        input_embeds = self.main_model.shared(input_ids)
        device = encoder_attention_mask.device
        B, _ = encoder_attention_mask.shape

        if self.has_graph:
            graph_rep = self.graph_encoder(batch)
            graph_rep = self.graph_projector(graph_rep)
            # graph_rep = torch.nn.functional.normalize(graph_rep, dim=1)
            input_embeds = torch.cat([graph_rep.unsqueeze(1), input_embeds[:, :-1, :]], dim=1)
            encoder_attention_mask = torch.cat([torch.ones(B, 1).to(device), encoder_attention_mask[:, :-1]], dim=1)
        
        loss = self.main_model(
            inputs_embeds = input_embeds,
            attention_mask=encoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=label # 有了labels就不用decoder_input_ids了，因为会自动将labels左移生成decoder_input_ids
            ).loss

        # print("loss: ", loss.item())

        return loss
    

    def translate(self, batch, input_ids, encoder_attention_mask, tokenizer):

        input_embeds = self.main_model.shared(input_ids)
        device = encoder_attention_mask.device
        B, _ = encoder_attention_mask.shape

        if self.has_graph:
            graph_rep = self.graph_encoder(batch)
            graph_rep = self.graph_projector(graph_rep)
            input_embeds = torch.cat([graph_rep.unsqueeze(1), input_embeds[:, :-1, :]], dim=1)
            encoder_attention_mask = torch.cat([torch.ones(B, 1).to(device), encoder_attention_mask[:, :-1]], dim=1)

        # input_prompt = "The molecule is"
        # input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
        # input_ids = input_ids.to(device)

        outputs = self.main_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=encoder_attention_mask,
            num_beams=5,
            # bos_token_id=102,
            max_length=512,
        )

        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return res



    # def forward(self, batch, input_ids, tgt_mask):
    #     device = input_ids.device
    #     typ = torch.zeros(input_ids.shape).long().to(device)

    #     node_reps, encoder_attention_mask = self.graph_encoder(batch)
    #     encoder_attention_mask = encoder_attention_mask.to(device)
    #     print("GIN output shape: ", node_reps.shape)
    #     print("T5 decoder input token shape: ", input_ids.shape)  # B,L
    #     print("T5 decoder input mask shape: ", tgt_mask.shape)  # B,L
    #     # print("GIN node mask: ", encoder_attention_mask[0])

    #     node_reps = self.graph_projector(node_reps)

    #     output = self.main_model(
    #         encoder_outputs=node_reps.unsqueeze(0), 
    #         decoder_input_ids=input_ids, 
    #         attention_mask=encoder_attention_mask,
    #         decoder_attention_mask=tgt_mask,
    #         )["last_hidden_state"]

    #     print("T5 decoder output shape: ", output.shape)

    #     return output
    
    def gin_encode(self, batch):
        node_reps = self.graph_encoder(batch)
        return node_reps


if __name__ == '__main__':
    model = T5ForConditionalGeneration.from_pretrained("molt5_base/")
    for p in model.named_parameters():
        if 'lm_head' in p[0] or 'shared' in p[0]:
	        print(p[1])
    
    print(model.shared)
    print(model.lm_head)