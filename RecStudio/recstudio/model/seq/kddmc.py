import torch
from recstudio.ann import sampler
from recstudio.data import dataset, advance_dataset
from recstudio.model import basemodel, loss_func, module, scorer

r"""
FPMC
#########

Paper Reference:
    Steffen Rendle, et al. "Factorizing personalized Markov chains for next-basket recommendation" in WWW2010.
    https://dl.acm.org/doi/10.1145/1772690.1772773
"""


class KDDMC(basemodel.BaseRetriever):
    r"""
    | FPMC is based on personalized transition graphs over underlying Markov chains. It
      factorizes the transition cube with a pairwise interaction model which is a special case of
      the Tucker Decomposition.
    """

    def _get_dataset_class():
        r"""The dataset FPMC used is SeqDataset."""
        return advance_dataset.KDDCUPSeqDataset
    
    def _set_data_field(self, data):
        data.use_field = set([data.fuid, data.fiid, data.frating, 'locale'])

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    def _get_query_encoder(self, train_data):

        class MCQueryEncoder(torch.nn.Module):
            
            def __init__(self, train_data, item_encoder) -> None:
                super().__init__()
                self.item_encoder = item_encoder
                self.fiid = train_data.fiid
            
            def forward(self, batch):
                item_seq = batch['in_' + self.fiid]
                seq_len = batch['seqlen'].unsqueeze(dim=-1) - 1
                last_item = torch.gather(item_seq, dim=-1, index=seq_len).squeeze()
                return self.item_encoder(last_item)
        
        return MCQueryEncoder(train_data, self.item_encoder)


    def _get_score_func(self):
        r"""Inner Product is used as the score function."""
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        r"""The loss function is BPR loss."""
        return loss_func.BPRLoss()

    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items)
