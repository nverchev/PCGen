python3 evaluate_samplings.py --dataset ShapenetFlow  --ae VQVAE --select_class car --dir_path /scratch  --exp car_with_EMD_dict_16_256 --decoder PCGenC --components 4 --c_reg 1  --gf --epochs 1000 --decay_period 900 --book_size 16 --cw_dim 256 --z_dim 16
Version  LDGCNN_PCGenCGF_Chamfer_VQVAE_car_with_EMD_dict_16_256
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 22/22 [00:02<00:00, 10.44it/s]
Metrics:
Chamfer: 5.8666e-04     Chamfer Augmented: 1.5630e-02   Chamfer Smooth: -2.6810e+07     EMD: 4.0494e-02
Minimum Matching Distance score (Chamfer):
Number of tests: 10 Min: 3.2686e-03 Max: 3.3276e-03 Mean: 3.2959e-03 Std: 1.7285e-05
Coverage score (Chamfer):
Number of tests: 10 Min: 4.6023e-01 Max: 4.8011e-01 Mean: 4.6903e-01 Std: 7.1250e-03
1-NNA score (Chamfer):
Number of tests: 10 Min: 5.6960e-01 Max: 5.8949e-01 Mean: 5.8125e-01 Std: 6.5956e-03


 python3  evaluate_samplings.py --dataset ShapenetFlow  --ae VQVAE --select_class chair --dir_path /scratch  --exp chair_with_EMD_dict_16_256 --decoder PCGenC --components 4 --c_reg 1  --gf --epochs 1000 --decay_period 100 --book_size 16 --cw_dim 256 --z_dim 16
Version  LDGCNN_PCGenCGF_Chamfer_VQVAE_chair_with_EMD_dict_16_256
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 42/42 [00:03<00:00, 12.53it/s]
Metrics:
Chamfer: 9.5328e-04     Chamfer Augmented: 1.8299e-02   Chamfer Smooth: -2.2023e+07     EMD: 4.7894e-02

Minimum Matching Distance score (Chamfer):
Number of tests: 10 Min: 6.7951e-03 Max: 7.0714e-03 Mean: 6.8897e-03 Std: 7.6879e-05
Coverage score (Chamfer):
Number of tests: 10 Min: 4.6073e-01 Max: 4.8792e-01 Mean: 4.7356e-01 Std: 8.7104e-03
1-NNA score (Chamfer):
Number of tests: 10 Min: 5.7100e-01 Max: 6.1405e-01 Mean: 6.0219e-01 Std: 1.1846e-02



python3 evaluate_samplings.py --dataset ShapenetFlow  --ae VQVAE --select_class airplane --dir_path /scratch  --exp airplane_with_EMD_dict_16_256 --decoder PCGenC --components 4 --c_reg 1  --gf --epochs 1000 --decay_period 900 --book_size 16 --cw_dim 256 --z_dim 16
Version  LDGCNN_PCGenCGF_Chamfer_VQVAE_airplane_with_EMD_dict_16_256
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [00:02<00:00, 11.08it/s]
Metrics:
Chamfer: 9.9249e-05     Chamfer Augmented: 6.3414e-03   Chamfer Smooth: -4.8071e+07     EMD: 2.2959e-02

Minimum Matching Distance score (Chamfer):
Number of tests: 10 Min: 8.2216e-04 Max: 8.8736e-04 Mean: 8.6104e-04 Std: 2.1699e-05
Coverage score (Chamfer):
Number of tests: 10 Min: 3.6049e-01 Max: 4.2222e-01 Mean: 3.9407e-01 Std: 1.5895e-02
1-NNA score (Chamfer):
Number of tests: 10 Min: 6.5185e-01 Max: 7.3210e-01 Mean: 6.7444e-01 Std: 2.3229e-02

Minimum Matching Distance score (Emd):
Number of tests: 10 Min: 5.7015e-02 Max: 5.9110e-02 Mean: 5.8006e-02 Std: 5.7193e-04
Coverage score (Emd):
Number of tests: 10 Min: 3.8765e-01 Max: 4.5926e-01 Mean: 4.1358e-01 Std: 2.0547e-02
1-NNA score (Emd):
Number of tests: 10 Min: 7.5926e-01 Max: 8.0617e-01 Mean: 7.7272e-01 Std: 1.4157e-02



Minimum Matching Distance score (Chamfer):
Number of tests: 10 Min: 8.1258e-04 Max: 9.1615e-04 Mean: 8.5643e-04 Std: 3.0181e-05
Coverage score (Chamfer):
Number of tests: 10 Min: 3.7037e-01 Max: 4.1728e-01 Mean: 3.9432e-01 Std: 1.4013e-02
1-NNA score (Chamfer):
Number of tests: 10 Min: 6.1605e-01 Max: 7.0494e-01 Mean: 6.5704e-01 Std: 2.6182e-02
Minimum Matching Distance score (Chamfer):
Number of tests: 10 Min: 8.1677e-04 Max: 8.9203e-04 Mean: 8.6556e-04 Std: 2.3084e-05
Coverage score (Chamfer):
Number of tests: 10 Min: 3.5309e-01 Max: 3.9506e-01 Mean: 3.7827e-01 Std: 1.3195e-02
1-NNA score (Chamfer):
Number of tests: 10 Min: 6.1481e-01 Max: 6.7160e-01 Mean: 6.4728e-01 Std: 1.6407e-02
Minimum Matching Distance score (Chamfer):
Number of tests: 10 Min: 8.3664e-04 Max: 8.8888e-04 Mean: 8.5510e-04 Std: 1.7436e-05
Coverage score (Chamfer):
Number of tests: 10 Min: 3.6543e-01 Max: 4.0741e-01 Mean: 3.8914e-01 Std: 1.4448e-02
1-NNA score (Chamfer):
Number of tests: 10 Min: 6.2099e-01 Max: 6.6667e-01 Mean: 6.4321e-01 Std: 1.5251e-02

Minimum Matching Distance score (Chamfer):
Number of tests: 10 Min: 8.0609e-04 Max: 8.8388e-04 Mean: 8.6058e-04 Std: 2.1564e-05
Coverage score (Chamfer):
Number of tests: 10 Min: 3.6790e-01 Max: 4.1728e-01 Mean: 3.8691e-01 Std: 1.3114e-02
1-NNA score (Chamfer):
Number of tests: 10 Min: 6.2469e-01 Max: 6.7778e-01 Mean: 6.4753e-01 Std: 1.8012e-02
