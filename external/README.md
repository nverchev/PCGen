Author: Nicolas Vercheval \
Date: 04/05/2023

## Earth Mover Distance implementations
Here are two commonly used implementations of the Earth Mode Distance (EMD) for point clouds.
Please refer to their documentation for more details.

### Installing the packages
The code to install the source code has been simplified and tested with cuda 11.8.
You can install the packages from their root folder with `pip install .`. 

### Troubleshooting
Pytorch comes with a version of cuda as a compute platform: https://pytorch.org/. 
Make sure that version matches your nvidia drivers.
You can check the version of your nvidia divers with  `nvcc --version`.

### Example use
These implementations, differently from the original ones, install the python modules as well. 
This way you can drop the relative path and use them in other projects as well.

```
# import torch first
import torch
from structural_losses import match_cost
from emd import emdModule


def random_point_cloud():
    return torch.rand(1, 2048, 3).to('cuda:0')


t1 = random_point_cloud()
t2 = random_point_cloud()
emd_dist = emdModule()
emd1 = torch.sqrt(emd_dist(t1, t2, 0.01, 200)[0]).mean(1)
emd2 = match_cost(t1, t2) / t1.shape[1]
print(emd1)
print(emd2)
```