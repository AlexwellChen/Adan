import torch
import pynvml
from typing import List

def get_param_info(params: List[List[torch.Tensor]]):
    if len(params) == 0:
        raise ValueError("params is empty")
    layer_info = [{} for _ in range(len(params[0]))]
    layer_num = len(params[0])
    for i in range(layer_num):
        layer_info[i]['index'] = i
        layer_size = 0 # total size of the layer, in bytes
        for param in params:
            layer_size += param[i].numel() * param[i].element_size()
        layer_info[i]['size'] = layer_size
    # sort layer_info by size, largest first
    layer_info.sort(key=lambda x: x['size'], reverse=True)
    return layer_info

def get_free_memory():
    pynvml.nvmlInit()
    # get current GPU
    idx = torch.cuda.current_device()
    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.free

def get_tensor_access_group(params: List[List[torch.Tensor]], ratio=None):
    '''
    Structure of params: [[layer1_tensor1, layer2_tensor1, ...], [layer1_tensor2, layer2_tensor2, ...], ...]
    Each layer's tensor will be accessed in a group
    Each group tensor size will be less than free memory * ratio, each group has at least one tensor

    '''
    # get layer info
    layer_info = get_param_info(params)
    '''
    layer_info structure:
    [
        {"layer_index": i, "layer_size": ...}
    ]
    '''

    # get free memory
    free_memory = get_free_memory()
    usable_memory = free_memory
    if ratio is not None:
        usable_memory = int(free_memory * ratio)
    # get tensor access group, maxium useage adapt to free memory
    tensor_access_group = []
    current_usage = 0
    added_layer = []
    for i in range(len(layer_info)):
        layer_idx = layer_info[i]['index']
        if current_usage + layer_info[i]['size'] < usable_memory:
            current_usage += layer_info[i]['size']
            added_layer.append(layer_idx)
        else:
            if len(added_layer) == 0:
                raise ValueError("No layer can be added to tensor_access_group, please increase ratio. Or it has reached the maximum usage.")
            temp_params = [[] for _ in range(len(params))]
            for i in range(len(params)):
                for layer in added_layer:
                    temp_params[i]=params[i][layer]
            tensor_access_group.append(temp_params)
            current_usage = 0
            added_layer = []
    # Save the last group
    if len(added_layer) != 0:
        temp_params = [[] for _ in range(len(params))]
        for i in range(len(params)):
                for layer in added_layer:
                    temp_params[i]=params[i][layer]
        tensor_access_group.append(temp_params)
    return tensor_access_group
    


                
            
        




    
    