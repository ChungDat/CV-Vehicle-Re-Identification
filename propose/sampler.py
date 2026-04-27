from torch.utils.data import Sampler
from collections import defaultdict
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly samples N identities, then for each identity,
    randomly samples K instances.

    Batch size = N * K
    """
    
    def __init__(self, dataset, batch_size, num_instances):
        """
        Args:
            dataset:
                dataset.samples should contain a list of tuples:
                (img_name, v_id, c_id)

            batch_size:
                total batch size

            num_instances:
                K images per identity
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances

        self.num_v_ids_per_batch = batch_size // num_instances

        # v_id -> list[index]
        self.index_dic = defaultdict(list)

        for index, (_, v_id, _, _) in enumerate(dataset.samples):
            self.index_dic[v_id].append(index) # append all indices of the same vehicle ID

        self.v_ids = list(self.index_dic.keys()) # all vehicle IDs

        self.length = 0

        for v_id in self.v_ids:
            idxs = self.index_dic[v_id] # list of all indices of the same vehicle ID

            num = len(idxs)

            if num < self.num_instances:
                num = self.num_instances

            self.length += num - num % self.num_instances # remove the last batch if it has fewer than num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        # prepare mini-batches for each v_id
        # {
        #   v_id1: [[idx1, idx2, ..., idxN], [idx1, idx2, ..., idxN], ...],
        #   v_id2: [[idx1, idx2, ..., idxN], [idx1, idx2, ..., idxN], ...],
        #   ...
        # }
        for v_id in self.v_ids:
            idxs = self.index_dic[v_id] # list of all indices of the same vehicle ID

            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs,
                    size=self.num_instances,
                    replace=True
                )

            idxs = list(idxs) # in case idxs is numpy array 

            random.shuffle(idxs)

            batch_idxs = []

            for idx in idxs:
                batch_idxs.append(idx)

                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[v_id].append(batch_idxs)
                    batch_idxs = []

        available_v_ids = list(self.v_ids) # copy list of v_ids 

        final_idxs = []

        while len(available_v_ids) >= self.num_v_ids_per_batch:
            selected_v_ids = random.sample(
                available_v_ids,
                self.num_v_ids_per_batch
            ) # random choose N v_ids 

            for v_id in selected_v_ids: # for each v_id, pop first batch_idxs 
                batch_idxs = batch_idxs_dict[v_id].pop(0)

                final_idxs.extend(batch_idxs) # final_idxs only stores indices (int), not v_ids 

                if len(batch_idxs_dict[v_id]) == 0: # if no more batch_idxs for this v_id, remove it from available_v_ids
                    available_v_ids.remove(v_id)

        return iter(final_idxs)

    def __len__(self):
        return self.length