import random

import numpy as np


def pooled_recall(queries, gallery, rows, levels, pool, n_pools, seed=1):
    if not 1 <= pool <= len(gallery):
        raise ValueError("pool must be between 1 and the gallery size")
    if n_pools < 1:
        raise ValueError("n_pools must be positive")
    rng = random.Random(seed)
    results = {}
    for level in levels:
        ridx = [i for i, row in enumerate(rows) if row[1] == level]
        if not ridx:
            continue
        hits, ranks = ({1: 0, 5: 0, 10: 0}, [])
        for _ in range(n_pools):
            ri = ridx[rng.randrange(len(ridx))]
            gold = rows[ri][0]
            cand = [gold] + rng.sample([i for i in range(len(gallery)) if i != gold], pool - 1)
            scores = (gallery[cand] @ queries[ri]).numpy()
            rank = int((scores[1:] > scores[0]).sum() + 0.5 * (scores[1:] == scores[0]).sum())
            ranks.append(rank + 1)
            for k in hits:
                hits[k] += rank < k
        results[level] = {f"R@{k}": hits[k] / n_pools for k in hits}
        results[level]["MedR"] = int(np.median(ranks))
    return results
