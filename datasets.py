import os
from typing import Tuple
import numpy as np
import torch
from collections import defaultdict
from models import KBCModel


class Dataset(object):
    def __init__(self, name: str):
        self.root = 'src_data/' + name + '/'

        # load entities and relations files
        with open(os.path.join(self.root, 'ent_id'), "r") as f:
            entities_to_id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
        with open(os.path.join(self.root, 'rel_id'), "r") as f:
            relations_to_id = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f.readlines()}
        print("{} entities and {} relations".format(len(entities_to_id), len(relations_to_id)))
        self.n_entities = len(entities_to_id)
        self.n_relations = len(relations_to_id)
        self.n_predicates = self.n_relations * 2

        # map train/test/valid with the ids
        files = ['train', 'valid', 'test']
        self.data = {}
        for f in files:
            file_path = os.path.join(self.root, f)
            to_read = open(file_path, 'r')
            examples = []
            for line in to_read.readlines():
                lhs, rel, rhs = line.strip().split('\t')
                try:
                    examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])
                except ValueError:
                    continue
            self.data[f] = np.array(examples).astype('uint64')

        # create filtering files
        to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
        for f in files:
            for lhs, rel, rhs in self.data[f]:
                to_skip['lhs'][(rhs, rel + self.n_relations)].add(lhs)  # reciprocals
                to_skip['rhs'][(lhs, rel)].add(rhs)

        self.to_skip = {'lhs': {}, 'rhs': {}}
        for kk, skip in to_skip.items():
            for k, v in skip.items():
                self.to_skip[kk][k] = sorted(list(v))

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
        return np.vstack((self.data['train'], copy))

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, device: str='cpu', missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10)
    ):
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).to(device)
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)

            mean_reciprocal_rank[m] = round(torch.mean(1. / ranks).item(), 3)
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))
        return mean_reciprocal_rank, hits_at

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities
