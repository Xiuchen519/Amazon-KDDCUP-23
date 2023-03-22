import torch
from recstudio.model.basemodel import BaseRanker
from recstudio.model.module import ctr, MLPModule
from recstudio.data.dataset import TripletDataset
from ..loss_func import BCEWithLogitLoss


class NFM(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        self.linear = ctr.LinearLayer(self.fields, train_data)
        self.fm = ctr.FMLayer()
        self.bn = torch.nn.BatchNorm1d(self.embed_dim)
        model_config = self.config['model']
        self.mlp = MLPModule([self.embed_dim]+model_config['mlp_layer']+[1],
                             activation_func=model_config['activation'],
                             dropout=model_config['dropout'],
                             batch_norm=model_config['batch_norm'],
                             last_activation=False, last_bn=False)

    def score(self, batch):
        linear_score = self.linear(batch)
        emb = self.embedding(batch)
        fm_emb = self.bn(self.fm(emb))
        mlp_score = self.mlp(fm_emb).squeeze(-1)
        return {'score' : linear_score + mlp_score}

    def _get_loss_func(self):
        return BCEWithLogitLoss()
