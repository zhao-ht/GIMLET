
from .model.contrastive_gin import GINSimclr
import torch
# {'graph': graph_batch,
#             'input_ids_pos': text_batch_pos.data['input_ids'],
#             'attention_mask_pos': text_batch_pos.data['attention_mask'],
#                'input_ids_neg': text_batch_neg.data['input_ids'],
#                'attention_mask_neg': text_batch_neg.data['attention_mask'],
#                'labels':labels_batch.data['labels']}

class Momu(GINSimclr):
    def forward(self,graph,input_ids_pos,attention_mask_pos,input_ids_neg,attention_mask_neg,labels=None):
        graph_rep = self.graph_encoder(graph)
        graph_rep = self.graph_proj_head(graph_rep)

        # print('graph')
        # print(graph_rep[:,0])
        text_rep_pos = self.text_encoder(input_ids_pos, attention_mask_pos)
        text_rep_pos = self.text_proj_head(text_rep_pos)

        text_rep_neg = self.text_encoder(input_ids_neg, attention_mask_neg)
        text_rep_neg = self.text_proj_head(text_rep_neg)
        # print('text')
        # print(text_rep[:,0])
        scores_pos = torch.cosine_similarity(
            graph_rep,
            text_rep_pos,dim=-1)
        scores_neg = torch.cosine_similarity(
            graph_rep,
            text_rep_neg, dim=-1)
        # print(scores1)
        # print(scores2)
        if labels is not None:
            loss=torch.nn.CrossEntropyLoss()(torch.cat([scores_neg.unsqueeze(1),scores_pos.unsqueeze(1)],1),labels.squeeze(1).long())
        else:
            loss=torch.zeros([1]).to(scores_pos.device)
        pred=torch.argmax(torch.cat([scores_pos.unsqueeze(-1),scores_neg.unsqueeze(-1)],-1),-1)

        return{'logits':{'pos':scores_pos,'neg':scores_neg},'pred':pred,'labels':labels,'loss':loss}


def get_MoMu_model(args):
    return Momu.load_from_checkpoint(args.init_checkpoint)

