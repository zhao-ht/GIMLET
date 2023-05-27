from .graphormer import GraphormerModel,GraphormerEncoder,base_architecture
import torch.nn as nn
from torch.nn import CrossEntropyLoss,MSELoss
import torch

class GraphormerModelMultiTask(GraphormerModel):
    @staticmethod
    def add_args(parser):
        parser=GraphormerModel.add_args(parser)

        parser.add_argument(
            "--cla_task_number",
            type=int,
        )

        parser.add_argument("--reg_task_number",type=int,)

        parser.add_argument("--cla_class_max_number",type=int,default=128)

        return parser
    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)

        # if not safe_hasattr(args, "max_nodes"):
        if not hasattr(args, "max_nodes"):
            args.max_nodes = args.tokens_per_sample

        # logger.info(args)

        encoder = GraphormerEncoderMultiTask(args)
        graphormer_model= cls(args, encoder)
        graphormer_model.restore_from_file(args)
        if  args.not_load_pretrained_model_output_layer:
            graphormer_model.encoder.reset_output_layer_parameters()
        return graphormer_model


    def restore_from_file(self,args):
        if args.restore_file_graphormer is not None:
            state = torch.load(args.restore_file_graphormer)
            if 'model' in state.keys():
                state=state['model']
            if args.not_load_pretrained_model_output_layer:
                state.pop('encoder.embed_cla_out.weight')
                state.pop('encoder.embed_cla_out.bias')
                state.pop('encoder.embed_reg_out.weight')
                state.pop('encoder.embed_reg_out.bias')
            missing_keys, unexpected_keys =self.load_state_dict(
                state, strict=False)
            print('missing_keys:', missing_keys)
            print('unexpected_keys:', unexpected_keys)
        else:
            print('no individual restore file of graphormer specified')

    def forward(self,  **kwargs):
        return self.encoder( **kwargs)










class GraphormerEncoderMultiTask(GraphormerEncoder):
    def __init__(self,args):
        super().__init__(args)

        self.cla_task_number=args.cla_task_number
        self.cla_class_max_number=args.cla_class_max_number
        self.reg_task_number=args.reg_task_number

        self.embed_cla_out = nn.Linear(
            args.encoder_embed_dim, args.cla_task_number*args.cla_class_max_number, bias=True
        ) #reshape output to vector of cla_task_number*cla_class_max_number dim

        self.embed_reg_out = nn.Linear(args.encoder_embed_dim,args.reg_task_number,bias=True)




    def reset_output_layer_parameters(self):
        self.embed_cla_out.reset_parameters()
        self.embed_reg_out.reset_parameters()
        print('output layers embed_cla_out and embed_reg_out are reinitialized')


    def forward(self, graph, label_cla,label_reg):

        graph=graph.to(self.embed_cla_out.weight.device)

        inner_states, graph_rep = self.graph_encoder(graph,perturb=None,)

        x = inner_states[-1].transpose(0, 1)[:,0,:]

        # project masked tokens only
        # if masked_tokens is not None:
        #     raise NotImplementedError

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # elif self.embed_out is not None:
        output_cla = self.embed_cla_out(x)

        output_reg = self.embed_reg_out(x)

        # output_cla batched_data.cla_label

        output_cla=output_cla.reshape([-1,self.cla_task_number,self.cla_class_max_number])

        cla_task_number_actual = label_cla.shape[1]
        if output_cla.shape[1]>cla_task_number_actual:
            output_cla=output_cla[:,0:cla_task_number_actual,:]

        output_cla_flat=output_cla.reshape([-1, self.cla_class_max_number])
        label_cla_flat=label_cla.reshape([-1])

        valid_ind_cla=label_cla!=-100
        if valid_ind_cla.sum()>0:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss_cla = loss_fct(output_cla_flat, label_cla_flat)
        else:
            loss_cla=torch.tensor(0.0).to(label_cla.device)

        reg_task_number_actual = label_reg.shape[1]
        if output_reg.shape[1]>reg_task_number_actual:
            output_reg=output_reg[:,0:reg_task_number_actual]

        valid_ind_reg = label_reg != -100
        if valid_ind_reg.sum()>0:
            loss_fct = MSELoss()
            loss_reg = loss_fct(output_reg[valid_ind_reg], label_reg[valid_ind_reg])
        else:
            loss_reg = torch.tensor(0.0).to(label_reg.device)

        if cla_task_number_actual+reg_task_number_actual>0:
            loss=(loss_cla*cla_task_number_actual+loss_reg*reg_task_number_actual)/(cla_task_number_actual+reg_task_number_actual)
        else:
            loss=torch.tensor(0.0).to(label_cla.device)
        return {'loss':loss,'output_cla':output_cla,'output_reg':output_reg}
