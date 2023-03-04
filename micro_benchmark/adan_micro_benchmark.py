import torch
import sys
sys.path.append('..')
from adan import Adan
import argparse
from torch.profiler import profile, record_function, ProfilerActivity

def get_fake_parameters(n_params=10, size=512):
    params = []
    for i in range(n_params):
        tensor = torch.randn(size, size, requires_grad=True, device='cpu')
        tensor.grad = torch.randn(size, size, device='cpu')
        params.append(tensor)
    return params

if __name__ == "__main__":
    prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/adan_micro_benchmark'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )

    params = get_fake_parameters(size=4096, n_params=10)
    optimizer = Adan(params=params, foreach=False, fused=False, adaptiv=True)
    with prof:
        for i in range(10):
            optimizer.step()
            prof.step()
    
    
        
