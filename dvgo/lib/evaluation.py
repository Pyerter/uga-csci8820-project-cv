import numpy as np
import scipy.signal
import torch


def rgb_ssim(
    img0,
    img1,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    """
    Calculate the SSIM (Structural Similarity Index) between two RGB images.

    Args:
        img0: First input image as an Numpy array.
        img1: Second input image as an Numpy array.
        max_val: The dynamic range of the pixel values (255 for 8-bit images).
        filter_size: Size of the Gaussian filter.
        filter_sigma: Standard deviation of the Gaussian filter.
        k1: Constant to stabilize division with weak denominator.
        k2: Constant to stabilize division with weak denominator.
        return_map: If True, return the SSIM map instead of average SSIM score.

    Returns:
        Average SSIM score or SSIM map depending on 'return_map' parameter.
    """
    assert (
        img0.shape == img1.shape and img0.shape[-1] == 3
    ), "Input images must have the same dimensions and three channels."

    # Construct a 1D Gaussian filter for blurring the images.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Define a function for 2D convolution.
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f[:, None], mode="valid")

    # Apply filter along each channel.
    filt_fn = lambda z: np.stack(
        [
            convolve2d(convolve2d(z[..., i], filt), filt[None, :])
            for i in range(z.shape[-1])
        ],
        -1,
    )
    mu0, mu1 = filt_fn(img0), filt_fn(img1)
    mu00, mu11, mu01 = mu0 * mu0, mu1 * mu1, mu0 * mu1
    sigma00, sigma11, sigma01 = (
        filt_fn(img0**2) - mu00,
        filt_fn(img1**2) - mu11,
        filt_fn(img0 * img1) - mu01,
    )

    # Ensure the variances are non-negative.
    sigma00, sigma11 = np.maximum(0.0, sigma00), np.maximum(0.0, sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))

    # Constants for SSIM calculation.
    c1, c2 = (k1 * max_val) ** 2, (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom

    # Return SSIM map or average SSIM value.
    return ssim_map if return_map else np.mean(ssim_map)


__LPIPS__ = {}


def init_lpips(net_name, device):
    """
    Initialize an LPIPS model for computing perceptual similarity.

    Args:
        net_name: Name of the network ('alex' or 'vgg').
        device: The torch device on which to operate.

    Returns:
        An LPIPS model object.
    """
    import lpips  # Import lpips at the point of use.

    print(f"Initializing LPIPS with network: {net_name}")
    return lpips.LPIPS(net=net_name, version="0.1").eval().to(device)


def rgb_lpips(np_gt, np_im, net_name, device):
    """
    Compute LPIPS between two images.

    Args:
        np_gt: Ground truth image as a Numpy array.
        np_im: Comparison image as a Numpy array.
        net_name: Network name ('alex' or 'vgg').
        device: The torch device on which to compute.

    Returns:
        Perceptual similarity score as a float.
    """
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute(2, 0, 1).to(device)
    im = torch.from_numpy(np_im).permute(2, 0, 1).to(device)
