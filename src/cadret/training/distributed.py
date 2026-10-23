import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x.contiguous())
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        g = torch.stack(grads)
        dist.all_reduce(g)
        return g[dist.get_rank()]


def gather_cat(x):
    return torch.cat(GatherLayer.apply(x), dim=0)


def gather_variable(tensor):
    sizes = [
        torch.zeros(1, dtype=torch.long, device=tensor.device) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(sizes, torch.tensor([len(tensor)], device=tensor.device))
    lengths = [int(size.item()) for size in sizes]
    padded = tensor.new_zeros((max(1, max(lengths)), *tensor.shape[1:]))
    padded[: len(tensor)] = tensor
    buffers = [torch.empty_like(padded) for _ in lengths]
    dist.all_gather(buffers, padded)
    return torch.cat([buffer[:length] for buffer, length in zip(buffers, lengths)], dim=0)
