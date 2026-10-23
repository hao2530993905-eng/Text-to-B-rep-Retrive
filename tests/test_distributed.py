import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cadret.training.distributed import gather_cat, gather_variable


def distributed_worker(rank, port):
    store = dist.TCPStore(
        "127.0.0.1", port, 2, False, timeout=timedelta(seconds=90), use_libuv=False
    )
    dist.init_process_group(
        "gloo", store=store, rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    try:
        local = torch.tensor([[float(rank + 1)]], requires_grad=True)
        gather_cat(local).square().sum().backward()
        torch.testing.assert_close(local.grad, torch.tensor([[4.0 * (rank + 1)]]))
        uneven = torch.empty(0, 1) if rank == 0 else torch.tensor([[1.0], [2.0]])
        torch.testing.assert_close(gather_variable(uneven), torch.tensor([[1.0], [2.0]]))
    finally:
        dist.destroy_process_group()


def gloo_device_available():
    if not dist.is_available() or not dist.is_gloo_available():
        return False
    try:
        dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")
    except RuntimeError:
        return False
    return True


@unittest.skipUnless(gloo_device_available(), "No usable Gloo network device on this host")
class DistributedTests(unittest.TestCase):
    def test_gradients_and_uneven_shards(self):
        store = dist.TCPStore(
            "127.0.0.1",
            0,
            2,
            True,
            timeout=timedelta(seconds=90),
            wait_for_workers=False,
            use_libuv=False,
        )
        mp.spawn(distributed_worker, args=(store.port,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
