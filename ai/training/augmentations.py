import random
import numpy as np
from typing import Tuple

class RetinalAugmentor:
    """
    Biomedical Data Augmentation Pipeline.
    Applies synchronized geometric transformations to both the image and the mask,
    while applying photometric perturbations strictly to the image to preserve label integrity.
    """
    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate90_prob: float = 0.5,
        brightness_range: Tuple[float, float] = (0.85, 1.15),
        contrast_range: Tuple[float, float] = (0.85, 1.15),
        gaussian_noise_std: float = 0.02
    ):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate90_prob = rotate90_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gaussian_noise_std = gaussian_noise_std

    def __call__(self, image: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image: numpy array [C, H, W] in [0, 1]
            masks: numpy array [M, H, W] binary [0, 1]
        """
        # 1. Horizontal Flip
        if random.random() < self.hflip_prob:
            image = np.flip(image, axis=2).copy()
            masks = np.flip(masks, axis=2).copy()

        # 2. Vertical Flip
        if random.random() < self.vflip_prob:
            image = np.flip(image, axis=1).copy()
            masks = np.flip(masks, axis=1).copy()

        # 3. Random Orthogonal Rotation (90, 180, 270 degrees)
        if random.random() < self.rotate90_prob:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            masks = np.rot90(masks, k=k, axes=(1, 2)).copy()

        # 4. Photometric Adjustments (Applied solely to image)
        # Brightness scaling
        alpha = random.uniform(self.brightness_range[0], self.brightness_range[1])
        image = image * alpha

        # Contrast scaling around mean
        beta = random.uniform(self.contrast_range[0], self.contrast_range[1])
        mean = np.mean(image, axis=(1, 2), keepdims=True)
        image = (image - mean) * beta + mean

        # Gaussian Noise
        if self.gaussian_noise_std > 0 and random.random() < 0.3:
            noise = np.random.normal(0, self.gaussian_noise_std, image.shape).astype(np.float32)
            image = image + noise

        # Clip image to valid pixel range [0, 1]
        image = np.clip(image, 0.0, 1.0)
        return image, masks
