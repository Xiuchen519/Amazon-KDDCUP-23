import argparse
import torch.distributed as dist # needed for Faiss for multi GPUs. 

from faiss_retriever import search_by_faiss


# cache used in xlm-roberta is "/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/.recstudio/cache/f30c8c1234f63ba5c60a8b7814777f60"

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--query_reps_path", type=str, default=None)
    parser.add_argument("--item_reps_path", type=str, default=None)
    parser.add_argument("--dataset_cache_path", type=str, default=None)
    parser.add_argument("--ranking_file", type=str, default=None)
    parser.add_argument("--use_gpu", action='store_true', default=False)
    parser.add_argument("--depth", type=int, default=350)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    search_by_faiss(args.query_reps_path, args.item_reps_path, args.dataset_cache_path, args.ranking_file, batch_size=512, depth=args.depth,
                    use_gpu=args.use_gpu)