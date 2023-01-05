python3 main.py --dataset ShapenetFlow  --ae VQVAE --select_class airplane --dir_path /scratch  --exp airplane  --z_dim 128 --decoder PCGenC --components 4 --c_reg 0.05
# Criterion: 5.1433e+02   Embed Loss: 3.6807e-02  Chamfer: 1.2766e+00     Chamfer Augmented: 1.6012e-02   KLD: 1.4689e+01 NLL: 4.9839e+02 Accuracy: 5.7466e-01
# Chamfer: 9.4483e-05     Chamfer Augmented: 6.2376e-03   Chamfer Smooth: -4.8173e+07     EMD: 8.4688e-04
python3 main.py --dataset ShapenetFlow  --ae VQVAE --select_class car --dir_path /scratch  --exp car  --z_dim 128 --decoder PCGenC --components 4 --c_reg 0.05
# Criterion: 5.7157e+02   Embed Loss: 8.6496e-02  Chamfer: 4.8590e+00     Chamfer Augmented: 3.1222e-02   KLD: 1.3665e+01 NLL: 5.5301e+02 Accuracy: 3.9784e-01
# Chamfer: 5.0134e-04     Chamfer Augmented: 1.4265e-02   Chamfer Smooth: -2.7980e+07     EMD: 6.8714e-04
python3 main.py --dataset ShapenetFlow  --ae VQVAE --select_class chair --dir_path /scratch  --exp chair  --z_dim 128 --decoder PCGenC --components 4 --c_reg 0.1
#Criterion: 5.4579e+02   Embed Loss: 1.0263e-01  Chamfer: 4.0969e+00     Chamfer Augmented: 2.7478e-02   KLD: 2.4592e+01 NLL: 5.1676e+02 Accuracy: 4.4586e-01
#Chamfer: 4.9640e-04     Chamfer Augmented: 1.3794e-02   Chamfer Smooth: -2.8449e+07     EMD: 7.8056e-04


python3 main.py --dataset ShapenetFlow  --ae VQVAE --select_class chair --dir_path /scratch  --exp chair  --z_dim 128 --decoder PCGenC --components 4 --c_reg 200000  --recon_loss ChamferS --load 0 --eval
# Criterion: -1.1591e+07  Embed Loss: 9.3644e-02  Chamfer: 4.4243e+00     Chamfer Augmented: 2.8759e-02   Chamfer Smooth: -1.1745e+07     KLD: 2.3125e+01 NLL: 5.0140e+02 Accuracy: 4.7943e-01
# Chamfer: 5.3566e-04     Chamfer Augmented: 1.4427e-02   Chamfer Smooth: -2.9155e+07     EMD: 6.9400e-04

python3 main.py --dataset ShapenetFlow  --ae VQVAE --select_class airplane --dir_path /scratch  --exp airplane_with_EMD --decoder PCGenC --components 4 --c_reg 0.3  --gf --epochs 800 --decay_period 700 --eval --load 0
# Chamfer: 8.1302e-05     Chamfer Augmented: 5.9487e-03   Chamfer Smooth: -5.0636e+07     EMD: 2.1367e-02

python3 main.py --dataset ShapenetFlow  --ae VQVAE --select_class airplane --dir_path /scratch  --exp airplane_with_EMD_dict_16 --decoder PCGenC --components 4 --c_reg 0.2  --gf --epochs 1000 --decay_period 900 --book_size 16
#Criterion: 1.0373e+02   Embed Loss: 6.9726e-02  Chamfer: 1.1012e+00     Chamfer Augmented: 1.5250e-02   EMD: 1.0242e+02
#Chamfer: 8.1631e-05     Chamfer Augmented: 5.9499e-03   Chamfer Smooth: -5.0557e+07     EMD: 2.1206e-02




 python3 main.py --dataset ShapenetFlow  --ae VQVAE --select_class car --dir_path /scratch  --exp car_with_EMD --decoder PCGenC --components 4 --c_reg 0.2  --gf --epochs 1000 --decay_period 900 --cw_dim 512 --eval --load 0
Criterion: 1.5595e+02   Embed Loss: 1.5821e-01  Chamfer: 4.4213e+00     Chamfer Augmented: 3.0604e-02   EMD: 1.5121e+02
Chamfer: 4.5633e-04     Chamfer Augmented: 1.3980e-02   Chamfer Smooth: -2.9697e+07     EMD: 3.5093e-02
