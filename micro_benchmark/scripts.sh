--------------benchmarking----------------
# Single tensor
python adan_micro_benchmark.py --foreach False --log_name Single_tensor
# Multiple tensors without adaptive
python adan_micro_benchmark.py --foreach True --log_name Multiple_tensors_without_adaptive
# Multiple tensors with adaptive, ratio = 0.5
python adan_micro_benchmark.py --foreach True --adaptive True --adaptive_ratio 0.5 --log_name Multiple_tensors_with_adaptive_ratio_05
# Multiple tensors with adaptive, ratio = 1.0
python adan_micro_benchmark.py --foreach True --adaptive True --adaptive_ratio 1.0 --log_name Multiple_tensors_with_adaptive_ratio_10
# Multiple tensors with adaptive, ratio = 0.25
python adan_micro_benchmark.py --foreach True --adaptive True --adaptive_ratio 0.25 --log_name Multiple_tensors_with_adaptive_ratio_025

# Single tensor fused
python adan_micro_benchmark.py --foreach False --fused True --log_name Single_tensor_fused
# Multiple tensors without adaptive fused
python adan_micro_benchmark.py --foreach True --fused True --log_name Multiple_tensors_without_adaptive_fused
# Multiple tensors with adaptive fused, ratio = 0.5
python adan_micro_benchmark.py --foreach True --fused True --adaptive True --adaptive_ratio 0.5 --log_name Multiple_tensors_with_adaptive_ratio_05_fused
# Multiple tensors with adaptive fused, ratio = 1.0
python adan_micro_benchmark.py --foreach True --fused True --adaptive True --adaptive_ratio 1.0 --log_name Multiple_tensors_with_adaptive_ratio_10_fused
# Multiple tensors with adaptive fused, ratio = 0.25
python adan_micro_benchmark.py --foreach True --fused True --adaptive True --adaptive_ratio 0.25 --log_name Multiple_tensors_with_adaptive_ratio_025_fused
--------------benchmarking end----------------