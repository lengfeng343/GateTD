import argparse
from typing import Dict

import torch
from torch import optim
from datasets import Dataset
from models import CP, GateTD, N3
from optimizers import KBCOptimizer


big_datasets = ['FB15K', 'WN', 'WN18RR', 'FB237']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)
parser.add_argument(
    '--dataset', choices=datasets, default='WN18RR',
    help="Dataset in {}".format(datasets)
)
models = ['CP', 'GateTD']
parser.add_argument(
    '--model', choices=models, default='GateTD',
    help="Model in {}".format(models)
)
gates = ['RNNCell', 'LSTMCell', 'GRUCell']
parser.add_argument(
    '--gate', choices=gates, default='GRUCell',
    help='Gating mechanism in {}'.format(gates)
)
parser.add_argument(
    '--regFlag', default=True,
    help='Whether to use N3 regularizer'
)
optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)
parser.add_argument(
    '--max_epochs', default=1500, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=5, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--batch_size', default=1024, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=3e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--edim', default=1000, type=int,
    help="FEntity embedding dimensionality."
)
parser.add_argument(
    '--rdim', default=500, type=int,
    help="Relation embedding dimensionality."
)
parser.add_argument(
    '--reg', default=5e-1, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
args = parser.parse_args()

dataset = Dataset(args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))
print(args.model)
print(dataset.get_shape())

model = {
    'CP': lambda: CP(dataset.get_shape(), args.edim, args.init),
    'GateTD': lambda: GateTD(dataset.get_shape(), args.edim, args.rdim, args.gate, args.init)
}[args.model]()

device = 'cpu'
model.to(device)

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate),
    'SparseAdam': lambda: optim.SparseAdam(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

optimizer = KBCOptimizer(model, args.regFlag, args.reg, optim_method, args.batch_size, device)


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}


cur_loss = 0
curve = {'train': [], 'valid': [], 'test': []}

for e in range(args.max_epochs):
    cur_loss = optimizer.epoch(e, examples)

    if (e + 1) % args.valid == 0:
        test = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000, device))
            for split in ['test']
        ]

        # curve['valid'].append(valid)
        # curve['train'].append(train)
        curve['test'].append(test)

        # print("\t TRAIN: ", train)
        print("\t epoch[", e, "], TEST : ", test)

results = dataset.eval(model, 'test', -1, device)
print("\n\nTEST : ", results)
