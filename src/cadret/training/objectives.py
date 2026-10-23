import numpy as np
import torch.nn.functional as F


def multipos(S, P):
    lq = F.log_softmax(S, dim=1)
    lc = F.log_softmax(S.t(), dim=1)
    Pt = P.t() / P.t().sum(1, keepdim=True).clamp(min=1e-06)
    return 0.5 * (-(P * lq).sum(1).mean() - (Pt * lc).sum(1).mean())


def eq_classes(V, thresh):
    seen, cls = ({}, np.zeros(len(V), dtype=np.int64))
    for i, k in enumerate((V / thresh).round().numpy().astype(np.int32)):
        cls[i] = seen.setdefault(k.tobytes(), len(seen))
    return (cls, len(seen))
