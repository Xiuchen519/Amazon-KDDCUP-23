import os
import dgl
import math
import torch
import pickle
import argparse
import pandas as pd

from tqdm import trange
import dgl.function as fn
from dgl.data import DGLDataset
from dgl.nn.pytorch.conv import SGConv

from co_occur_graph import CoOccuGraph


class KDDCupGraphDataset(DGLDataset):
    def __init__(self):
        self.my_data_dir = '/root/autodl-tmp/huangxu/Amazon-KDDCUP-23/co-occurrence-graph/'
        super().__init__(name='kdd_cup_graph')

    
    def process(self):

        return super().process()



def load_graph(fpath: str) -> dgl.DGLGraph:
    co_graph = CoOccuGraph({'a':1})
    co_graph.load(fpath)
    g = dgl.from_scipy(co_graph.graph)
    return g, co_graph.item_map


def construct_negative_graph(graph, k):
    src, dst = graph.edges()
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,), device=graph.device)
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())


class DotProductPredictor(torch.nn.Module):
    def forward(self, graph, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


class SGCModel(torch.nn.Module):
    def __init__(self, num_items, embed_dim, K=2):
        super(SGCModel, self).__init__()
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.embed_dim = embed_dim
        self.K = K
        self.conv = SGConv(self.embed_dim, self.embed_dim, k=K, allow_zero_in_degree=True)
        self.pred = DotProductPredictor()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.item_embedding.weight.data.uniform_(-stdv, stdv)

    def forward(self, g, neg_g):
        x = self.item_embedding.weight
        h = self.conv(g, x)
        return self.pred(g, h), self.pred(neg_g, h)

    def get_node_embedding(self, g):
        x = self.item_embedding.weight
        h = self.conv(g, x)
        return h


def compute_loss(pos_score, neg_score):
    # 间隔损失
    n_edges = pos_score.shape[0]
    return (1 - pos_score + neg_score.view(n_edges, -1)).clamp(min=0).mean()


def save_embedding(item_map, model, g, fname):
    node_embedding = model.get_node_embedding(g)
    emb_file = {
        'embedding': node_embedding.cpu().data,
        'map': item_map
    }
    with open(fname, 'wb') as f:
        pickle.dump(emb_file, f)
    return



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_neg', type=int, default=3)
    parser.add_argument('--n_hop', type=int, default=2)
    parser.add_argument('--save', type=str, default="/root/autodl-tmp/huangxu/Amazon-KDDCUP-23/co-occurrence_graph/graph_emb.pkl")
    args, _ = parser.parse_known_args()

    graph_file = '/root/autodl-tmp/huangxu/Amazon-KDDCUP-23/co-occurrence_graph/graph_all.gph'
    graph, item_map = load_graph(graph_file)

    model = SGCModel(graph.num_nodes(), args.embed_dim, args.n_hop)
    device = torch.device('cuda:0')
    model = model.to(device)
    graph = graph.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    try:
        for epoch in range(args.epoch):
            negative_graph = construct_negative_graph(graph, args.n_neg)
            pos_score, neg_score = model(graph, negative_graph)
            loss = compute_loss(pos_score, neg_score)
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f"Epoch={epoch}: loss = {loss.item()}")
    except KeyboardInterrupt:
        pass

    print('Training End.')

    save_embedding(item_map, model, graph, args.save)

    print(f"Embedding saved in {args.save}.")
    print("End.") 

