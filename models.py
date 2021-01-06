from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch


class KBCModel(torch.nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]

                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks


class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.lhs = torch.nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = torch.nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = torch.nn.Embedding(sizes[2], rank, sparse=True)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

        self.input_dropout = torch.nn.Dropout(0.1)
        self.hidden_dropout = torch.nn.Dropout(0.1)

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        w_mat = lhs * rel
        pred = w_mat @ self.rhs.weight.t()
        return pred, (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data


class GateTD(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], edim: int, rdim: int, gatecell: str,
            init_size: float = 1e-3,
    ):
        super(GateTD, self).__init__()
        self.sizes = sizes
        self.edim = edim
        self.rdim = rdim
        self.gatecell = gatecell

        self.lhs = torch.nn.Embedding(sizes[0], edim, sparse=True)
        self.rel = torch.nn.Embedding(sizes[1], rdim, sparse=True)
        self.rhs = torch.nn.Embedding(sizes[2], edim, sparse=True)

        self.gate = {
            'RNNCell': lambda: torch.nn.RNNCell(rdim, edim),
            'LSTMCell': lambda: torch.nn.LSTMCell(rdim, edim),
            'GRUCell': lambda: torch.nn.GRUCell(rdim, edim)
        }[gatecell]()

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        if self.gatecell == 'LSTMCell':
            c = torch.zeros_like(lhs)
            rel_update, c1 = self.gate(rel, (lhs, c))
        else:
            rel_update = self.gate(rel, lhs)

        return torch.sum(lhs * rel_update * rhs, 1, keepdim=True)

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        if self.gatecell == 'LSTMCell':
            c = torch.zeros_like(lhs)
            rel_update, c1 = self.gate(rel, (lhs, c))
        else:
            rel_update = self.gate(rel, lhs)
        output = lhs * rel_update
        pred = output @ self.rhs.weight.t()
        return pred, (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.lhs(queries[:, 0])
        rel = self.rel(queries[:, 1])

        if self.gatecell == 'LSTMCell':
            c = torch.zeros_like(lhs)
            rel_update, c1 = self.gate(rel, (lhs, c))
        else:
            rel_update = self.gate(rel, lhs)
        return lhs * rel_update


class N3(torch.nn.Module):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight        # 0.01

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0]
