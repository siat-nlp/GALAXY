"""
DataLoader class
"""

import math

from galaxy.args import str2bool
from galaxy.data.batch import batch
from galaxy.data.sampler import RandomSampler
from galaxy.data.sampler import SequentialSampler
from galaxy.data.sampler import SortedSampler


class DataLoader(object):
    """ Implement of DataLoader. """

    @classmethod
    def add_cmdline_argument(cls, group):
        group.add_argument("--shuffle", type=str2bool, default=True)
        group.add_argument("--sort_pool_size", type=int, default=0)
        return group

    def __init__(self, dataset, hparams, collate_fn=None, sampler=None, is_test=False):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.sort_pool_size = hparams.sort_pool_size

        if sampler is None:
            if hparams.shuffle and not is_test:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        if self.sort_pool_size > 0 and not is_test:
            sampler = SortedSampler(sampler, self.sort_pool_size)

        def reader():
            for idx in sampler:
                yield idx

        self.reader = batch(reader, batch_size=hparams.batch_size, drop_last=False)
        self.num_batches = math.ceil(len(dataset) / hparams.batch_size)

        return

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        """
        1. Sampler -> batch data index：[1, 2, 3]
        2. Dataset -> batch data：[[x1, y1], [x2, y2], [x3, y3]]
        3. collate_fn -> batch data: [[x1, x2, x3], [y1, y2, y3]]
        """
        for batch_indices in self.reader():
            samples = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(samples)
