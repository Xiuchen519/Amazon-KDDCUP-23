import torch
from recstudio.data.dataset import UserDataset
from recstudio.model.basemodel import BaseRetriever, Recommender
from recstudio.model.loss_func import SoftmaxLoss
from recstudio.model.module import MLPModule
from recstudio.model.scorer import InnerProductScorer


class MultiDAEQueryEncoder(torch.nn.Module):
    def __init__(self, fiid, num_items, embed_dim, dropout_rate,
                 encoder_dims, decoder_dims, activation='relu'):
        super().__init__()
        assert encoder_dims[-1] == decoder_dims[0], 'expecting the output size of'\
            'encoder is equal to the input size of decoder.'
        assert encoder_dims[0] == decoder_dims[-1], 'expecting the output size of'\
            'decoder is equal to the input size of encoder.'

        self.fiid = fiid
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim, 0)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.encoder_decoder = torch.nn.Sequential(
            MLPModule([embed_dim]+encoder_dims+decoder_dims[1:], activation),
            torch.nn.Linear(decoder_dims[-1], embed_dim)
        )

    def forward(self, batch):
        # encode
        seq_emb = self.item_embedding(batch["in_"+self.fiid])
        non_zero_num = batch["in_"+self.fiid].count_nonzero(dim=1).unsqueeze(-1)
        seq_emb = seq_emb.sum(1) / non_zero_num.pow(0.5)
        h = self.dropout(seq_emb)

        return self.encoder_decoder(h)


class MultiDAE(BaseRetriever):

    # def add_model_specific_args(parent_parser):
    #     parent_parser = Recommender.add_model_specific_args(parent_parser)
    #     parent_parser.add_argument_group('MultiDAE')
    #     parent_parser.add_argument("--dropout", type=int, default=0.5, help='dropout rate for MLP layers')
    #     parent_parser.add_argument("--encoder_dims", type=int, nargs='+', default=64, help='MLP layer size for encoder')
    #     parent_parser.add_argument("--decoder_dims", type=int, nargs='+', default=64, help='MLP layer size for decocer')
    #     parent_parser.add_argument("--activation", type=str, default='relu', help='activation function for MLP layers')
    #     return parent_parser

    def _get_dataset_class():
        return UserDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, 0)

    def _get_query_encoder(self, train_data):
        model_config = self.config['model']
        return MultiDAEQueryEncoder(train_data.fiid, train_data.num_items,
                                    self.embed_dim, model_config['dropout'], model_config['encoder_dims'],
                                    model_config['decoder_dims'], model_config['activation'])

    def _get_score_func(self):
        return InnerProductScorer()

    def _get_sampler(self, train_data):
        return None

    def _get_loss_func(self):
        return SoftmaxLoss()
