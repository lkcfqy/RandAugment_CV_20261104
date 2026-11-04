# Copyright (c) OpenMMLab. All rights reserved. Apache-2.0.
# Test-only pure method extraction; not part of the upload payload.
import cv2, numpy as np
from typing import Optional
from types import SimpleNamespace
_MAX_LEVEL=10
imread_backend='cv2'
cache_randomness=lambda f:f
TRANSFORMS=SimpleNamespace(register_module=lambda:lambda cls:cls)
class BaseTransform:
 def __call__(self,results):return self.transform(results)
def bgr2gray(img):return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def level_to_mag(level: Optional[int], min_mag: float,
                 max_mag: float) -> float:
    """Map from level to magnitude."""
    if level is None:
        return round(np.random.rand() * (max_mag - min_mag) + min_mag, 1)
    else:
        return round(level / _MAX_LEVEL * (max_mag - min_mag) + min_mag, 1)

def adjust_color(img, alpha=1, beta=None, gamma=0, backend=None):
    r"""It blends the source image and its gray image:

    .. math::
        output = img * alpha + gray\_img * beta + gamma

    Args:
        img (ndarray): The input source image.
        alpha (int | float): Weight for the source image. Default 1.
        beta (int | float): Weight for the converted gray image.
            If None, it's assigned the value (1 - `alpha`).
        gamma (int | float): Scalar added to each sum.
            Same as :func:`cv2.addWeighted`. Default 0.
        backend (str | None): The image processing backend type. Options are
            `cv2`, `pillow`, `None`. If backend is None, the global
            ``imread_backend`` specified by ``mmcv.use_backend()`` will be
            used. Defaults to None.

    Returns:
        ndarray: Colored image which has the same size and dtype as input.
    """
    if backend is None:
        backend = imread_backend
    if backend not in ['cv2', 'pillow']:
        raise ValueError(f'backend: {backend} is not supported.'
                         f"Supported backends are 'cv2', 'pillow'")

    if backend == 'pillow':
        assert img.dtype == np.uint8, 'Pillow backend only support uint8 type'
        warnings.warn("Only use 'alpha' for pillow backend.")
        # Image.fromarray defaultly supports RGB, not BGR.
        pil_image = Image.fromarray(img[..., ::-1], mode='RGB')
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(alpha)
        return np.array(pil_image, dtype=img.dtype)[..., ::-1]
    else:
        gray_img = bgr2gray(img)
        gray_img = np.tile(gray_img[..., None], [1, 1, 3])
        if beta is None:
            beta = 1 - alpha
        colored_img = cv2.addWeighted(img, alpha, gray_img, beta, gamma)
        if not colored_img.dtype == np.uint8:
            # Note when the dtype of `img` is not the default `np.uint8`
            # (e.g. np.float32), the value in `colored_img` got from cv2
            # is not guaranteed to be in range [0, 255], so here clip
            # is needed.
            colored_img = np.clip(colored_img, 0, 255)
        return colored_img.astype(img.dtype)

mmcv=SimpleNamespace(adjust_color=adjust_color)

class ColorTransform(BaseTransform):
    """Base class for color transformations. All color transformations need to
    inherit from this base class. ``ColorTransform`` unifies the class
    attributes and class functions of color transformations (Color, Brightness,
    Contrast, Sharpness, Solarize, SolarizeAdd, Equalize, AutoContrast, Invert,
    and Posterize), and only distort color channels, without impacting the
    locations of the instances.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing the geometric
            transformation and should be in range [0, 1]. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for color transformation.
            Defaults to 0.1.
        max_mag (float): The maximum magnitude for color transformation.
            Defaults to 1.9.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.1,
                 max_mag: float = 1.9) -> None:
        assert 0 <= prob <= 1.0, f'The probability of the transformation ' \
                                 f'should be in range [0,1], got {prob}.'
        assert level is None or isinstance(level, int), \
            f'The level should be None or type int, got {type(level)}.'
        assert level is None or 0 <= level <= _MAX_LEVEL, \
            f'The level should be in range [0,{_MAX_LEVEL}], got {level}.'
        assert isinstance(min_mag, float), \
            f'min_mag should be type float, got {type(min_mag)}.'
        assert isinstance(max_mag, float), \
            f'max_mag should be type float, got {type(max_mag)}.'
        assert min_mag <= max_mag, \
            f'min_mag should smaller than max_mag, ' \
            f'got min_mag={min_mag} and max_mag={max_mag}'
        self.prob = prob
        self.level = level
        self.min_mag = min_mag
        self.max_mag = max_mag

    def _transform_img(self, results: dict, mag: float) -> None:
        """Transform the image."""
        pass

    @cache_randomness
    def _random_disable(self):
        """Randomly disable the transform."""
        return np.random.rand() > self.prob

    @cache_randomness
    def _get_mag(self):
        """Get the magnitude of the transform."""
        return level_to_mag(self.level, self.min_mag, self.max_mag)

    def transform(self, results: dict) -> dict:
        """Transform function for images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Transformed results.
        """

        if self._random_disable():
            return results
        mag = self._get_mag()
        self._transform_img(results, mag)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'level={self.level}, '
        repr_str += f'min_mag={self.min_mag}, '
        repr_str += f'max_mag={self.max_mag})'
        return repr_str

class Color(ColorTransform):
    """Adjust the color balance of the image, in a manner similar to the
    controls on a colour TV set. A magnitude=0 gives a black & white image,
    whereas magnitude=1 gives the original image. The bboxes, masks and
    segmentations are not modified.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Color transformation.
            Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Color transformation.
            Defaults to 0.1.
        max_mag (float): The maximum magnitude for Color transformation.
            Defaults to 1.9.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.1,
                 max_mag: float = 1.9) -> None:
        assert 0. <= min_mag <= 2.0, \
            f'min_mag for Color should be in range [0,2], got {min_mag}.'
        assert 0. <= max_mag <= 2.0, \
            f'max_mag for Color should be in range [0,2], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, results: dict, mag: float) -> None:
        """Apply Color transformation to image."""
        # NOTE defaultly the image should be BGR format
        img = results['img']
        results['img'] = mmcv.adjust_color(img, mag).astype(img.dtype)
