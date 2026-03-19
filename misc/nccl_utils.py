import torch
from torch.distributed.distributed_c10d import (
        Backend, PrefixStore, _new_process_group_helper, _world, rendezvous
    )
from datetime import timedelta

def create_nccl_process_group(init_method, rank, world_size, group_name, timeout_seconds):
    '''
        Create an NCCL process group for weight broadcast between training rank 0
        and vllm rollout workers. We can't reuse ds's or vllm's groups because
        neither spans both training and rollout participants. We use pytorch internals
        (_new_process_group_helper) instead of init_process_group() to avoid overwriting
        the default process group that ds and vllm depend on.
    '''
    timeout = timedelta(seconds=timeout_seconds)

    # All participants connect to a shared TCP store and wait for world_size peers
    rendezvous_iterator = rendezvous(url=init_method, rank=rank, world_size=world_size, timeout=timeout)
    # Rendezvous may reassign rank/world_size (e.g. workers join or leave dynamically), so we use its return values.
    # In our case, they match the inputs because the group has fixed membership as training rank 0 + N vllm workers, set once at init.
    store, rank, world_size = next(rendezvous_iterator)
    store.set_timeout(timeout)

    # Namespace the store so keys don't collide with ds/vllm groups
    store = PrefixStore(prefix=group_name, store=store)

    # pytorch 2.6+ renamed pg_options to backend_options
    kwargs = {}
    torch_version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
    if torch_version >= (2, 6):
        kwargs["backend_options"] = None

    else:
        kwargs["pg_options"] = None

    # Create the group without overwriting the default process group
    pg, _ = _new_process_group_helper(world_size=world_size,
                                      rank=rank,
                                      ranks=[],
                                      backend=Backend("nccl"),
                                      store=store,
                                      group_name=group_name,
                                      timeout=timeout,
                                      **kwargs
                                      )

    # Register rank mapping so torch.distributed.broadcast(..., group=pg) can
    # resolve rank numbers within this group
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg
