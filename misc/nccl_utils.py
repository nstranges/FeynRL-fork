from torch.distributed.distributed_c10d import (
Backend, PrefixStore, _new_process_group_helper, _world
    )
from torch.distributed import TCPStore
from datetime import timedelta
import ray

def create_nccl_process_group(init_method, rank, world_size, group_name, timeout_seconds, backend="nccl"):
    '''
        Create a process group for weight broadcast between training rank 0
        and vllm rollout workers. We can't reuse ds's or vllm's groups because
        neither spans both training and rollout participants. We use pytorch internals
        (_new_process_group_helper) instead of init_process_group() to avoid overwriting
        the default process group that ds and vllm depend on.
        backend: nccl for gpu-to-gpu broadcast, gloo for cpu-based broadcast.
    '''
    timeout = timedelta(seconds=timeout_seconds)

    # Parse host and port from tcp:// init_method
    # init_method format: "tcp://host:port"
    addr = init_method.replace("tcp://", "")
    host, port = addr.rsplit(":", 1)
    port = int(port)

    # Rank 0 creates the TCP store server; others connect as clients.
    store = TCPStore(host_name=host,
                     port=port,
                     world_size=world_size,
                     is_master=(rank == 0),
                     timeout=timeout,
                     wait_for_workers=True,
                     )

    # Namespace the store so keys don't collide with ds/vllm groups
    store = PrefixStore(prefix=group_name, store=store)

    # nccl backend requires ProcessGroupNCCL.Options for proper GPU communicator init
    # across separate Ray actor processes.
    pg_options = None
    if str(backend) == "nccl":
        from torch.distributed.distributed_c10d import ProcessGroupNCCL
        pg_options = ProcessGroupNCCL.Options()
        pg_options.is_high_priority_stream = False

    # Create the group without overwriting the default process group.
    # PyTorch 2.7+ renamed: world_size→group_size, rank→group_rank, ranks→global_ranks_in_group
    pg, _ = _new_process_group_helper(group_size=world_size,
                                      group_rank=rank,
                                      global_ranks_in_group=list(range(world_size)),
                                      backend=Backend(backend),
                                      store=store,
                                      group_name=group_name,
                                      timeout=timeout,
                                      backend_options=pg_options,
                                      )

    # Register rank mapping so torch.distributed.broadcast(..., group=pg) can
    # resolve rank numbers within this group
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg

@ray.remote
class NCCLBarrier:
    '''
        Lightweight barrier for synchronizing nccl broadcast participants.
        Each rollout engine signals ready from inside receive_all_weights_nccl
        BEFORE entering the NCCL broadcast loop. The driver polls get_count()
        and only fires the training broadcast once all engines have signaled.
        This replaces the unreliable time.sleep() approach.
    '''
    def __init__(self, expected):
        self.expected = expected
        self.count = 0

    def signal_ready(self):
        self.count += 1
        return self.count

    def get_count(self):
        return self.count