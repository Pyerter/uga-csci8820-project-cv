## TensoRF
Testing done with NVidia 1070.  
Batch sizes were 4096.  

The PSNR and SSIM scores were unusually high, so I believe there's something wrong with my implementation of their TensoRF framework, although I haven't been able to find where the failure is at.  

### CP Vectorization
10k Iterations  

| Folder | Train PSNR | Test PSNR | Test SSIM |
| Chair | 27.98 | 56.62 | 0.9969 |
| Drums | 49.67 | 60.44 | 0.9958 |

### VM Vectorization
20k Iterations, Batch Size 4096, Density Components [16, 16, 16], Feature Components [24, 24, 24] Appearance Dimensionality 27, Number Features 128.  

| Folder    | Train PSNR | Test PSNR | Test SSIM | Train Time | Test Time | Weights Size |
| Chair     | 27.41      | 54.49     | 0.9951    | 12:00      | 10:46     |              |
| Drums     | 44.78      | 57.95     | 0.9915    | 13:06      | 10:30     | 4.99MB       |

