from setuptools import setup, Command
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.cuda import is_available
import sys

cuda_extension = CUDAExtension(
            'fused_adan', 
            sources=['fused_adan/pybind_adan.cpp','./fused_adan/fused_adan_kernel.cu', './fused_adan/multi_tensor_adan_kernel.cu']
        )

ext_list = [cuda_extension]

if "--unfused" in sys.argv:
    print("Building unfused version of adan")
    ext_list = []
    sys.argv.remove("--unfused")

setup(
    name='adan',
    python_requires='>=3.8',
    version='0.0.1',
    install_requires=['torch'],
    py_modules=['adan'],
    description=(
        'Adan: Adaptive Nesterov Momentum Algorithm for '
        'Faster Optimizing Deep Models'
    ),
    author=(
        'Xie, Xingyu and Zhou, Pan and Li, Huan and '
        'Lin, Zhouchen and Yan, Shuicheng'
    ),
    ext_modules=ext_list if is_available() else [],
    cmdclass={'build_ext': BuildExtension} if is_available() else {},
)
