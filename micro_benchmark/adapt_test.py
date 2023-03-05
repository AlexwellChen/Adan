import sys
sys.path.append('..')
from adapt_tensor_access import adapt
import torch

if __name__ == "__main__":
    params = []
    for i in range(30):
        tensor = torch.randn(4096, 4096, requires_grad=True, device='cuda')
        tensor.grad = torch.randn(4096, 4096, device='cuda')
        params.append(tensor)
    tensor_access_group = adapt.get_tensor_access_group([params], ratio=0.5)
    print("Layer info: ", adapt.get_param_info([params]))
    
    print(len(tensor_access_group))