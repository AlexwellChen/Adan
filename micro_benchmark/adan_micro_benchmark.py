import torch
import sys
sys.path.append('..')
from adan import Adan
import argparse
from torch.profiler import profile, record_function, ProfilerActivity

def get_fake_parameters(n_params=10, size=512):
    params = []
    for i in range(n_params):
        tensor = torch.randn(size, size, requires_grad=True, device='cuda')
        tensor.grad = torch.randn(size, size, device='cuda')
        params.append(tensor)
    return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # foreach, default is False
    parser.add_argument('--foreach', type=bool, default=False)
    # fused, default is False
    parser.add_argument('--fused', type=bool, default=False)
    # adaptive, default is False
    parser.add_argument('--adaptive', type=bool, default=False)
    # adaptive_ratio, default is None
    parser.add_argument('--adaptive_ratio', type=float, default=None)
    # log name
    parser.add_argument('--log_name', type=str, default='adan_micro_benchmark')
    args = parser.parse_args()

    prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/'+args.log_name),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )

    params = get_fake_parameters(size=4096, n_params=30)
    optimizer = Adan(params=params, foreach=args.foreach, fused=args.fused, adaptive=args.adaptive, adaptive_ratio=args.adaptive_ratio)
    with prof:
        for i in range(10):
            optimizer.step()
            prof.step()
    
    
        
