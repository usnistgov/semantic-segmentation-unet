# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import numpy as np
import scipy
import scipy.ndimage
import scipy.signal
import skimage.io
import skimage.transform


import random
import skimage.transform
import torch


from albumentations.core.transforms_interface import DualTransform, to_tuple, BasicTransform


def apply_affine_transformation(I, angle, reflect_x, reflect_y, jitter_x, jitter_y, scale_x, scale_y):

    if angle != 0:
        I = skimage.transform.rotate(I, angle, preserve_range=True, mode='reflect')

    tform = skimage.transform.AffineTransform(translation=(jitter_x, jitter_y),
                                              scale=(scale_x, scale_y))
    I = skimage.transform.warp(I, tform._inv_matrix, mode='reflect', preserve_range=True)

    if reflect_x:
        I = np.fliplr(I)
    if reflect_y:
        I = np.flipud(I)

    return I.copy()  # avoid potential stride issues with numpy


class RotationTransform(DualTransform):
    """Randomly apply rotation transforms.

    Args:
        rotate_limit ((int, int) or int): rotation range. If rotate_limit is a single int value, the
            range will be (-rotate_limit, rotate_limit). Default: (-45, 45).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        rotate_limit=360,
            always_apply=False,
            p=0.5
    ):
        super(RotationTransform, self).__init__(always_apply, p)
        self.rotate_limit = to_tuple(rotate_limit)

    def apply(self, img, angle=0, **params):

        if angle != 0:
            img = skimage.transform.rotate(img, angle, preserve_range=True, mode='reflect')

        return img

    def apply_to_mask(self, img, angle=0, **params):
        if angle != 0:
            img = skimage.transform.rotate(img, angle, preserve_range=True, mode='reflect')
        return img

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError("Method apply_to_bbox is not implemented in class " + self.__class__.__name__)

    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError("Method apply_to_keypoint is not implemented in class " + self.__class__.__name__)

    def apply_to_bboxes(self, bboxes, **params):
        raise NotImplementedError("Method apply_to_bboxes is not implemented in class " + self.__class__.__name__)

    def apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError("Method apply_to_keypoints is not implemented in class " + self.__class__.__name__)

    def get_params(self):
        return {
            "angle": random.uniform(self.rotate_limit[0], self.rotate_limit[1]),
        }

    def get_transform_init_args(self):
        return {
            "rotate_limit": self.rotate_limit,
        }


class BlurTransform(DualTransform):
    """Randomly apply an image blur transforms.

    Args:
        blur_augmentation_max_sigma (int): maximum blur kernel sigma.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        blur_augmentation_max_sigma=0,
            always_apply=False,
            p=0.5
    ):
        super(BlurTransform, self).__init__(always_apply, p)
        self.blur_augmentation_max_sigma = blur_augmentation_max_sigma

    def apply(self, img, blur_sigma=0, **params):
        if blur_sigma > 0:
            img = scipy.ndimage.filters.gaussian_filter(img, blur_sigma, mode='reflect')

        return img

    def apply_to_mask(self, img, blur_sigma=0, **params):
        return img

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError("Method apply_to_bbox is not implemented in class " + self.__class__.__name__)

    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError("Method apply_to_keypoint is not implemented in class " + self.__class__.__name__)

    def apply_to_bboxes(self, bboxes, **params):
        raise NotImplementedError("Method apply_to_bboxes is not implemented in class " + self.__class__.__name__)

    def apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError("Method apply_to_keypoints is not implemented in class " + self.__class__.__name__)

    def get_params(self):
        return {
            "blur_sigma": random.uniform(0, self.blur_augmentation_max_sigma),
        }

    def get_transform_init_args(self):
        return {
            "blur_augmentation_max_sigma": self.blur_augmentation_max_sigma,
        }







class MajurskiAugment(DualTransform):
    """Randomly apply a blend of common transforms.

    Args:
        rotation_flag (bool): whether to apply rotation or not.
        reflection_flag (bool): whether to apply rotation or not.
        jitter_augmentation_severity (float): jitter augmentation severity as a fraction of the image size.
        scale_augmentation_severity (float): scale augmentation as a percentage of the image size.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(self, rotation_flag=False, reflection_flag=False, jitter_augmentation_severity=0, scale_augmentation_severity=0, always_apply=False, p=1.0):
        super(MajurskiAugment, self).__init__(always_apply, p)
        # convert input Nones to expected
        if jitter_augmentation_severity is None:
            jitter_augmentation_severity = 0
        if scale_augmentation_severity is None:
            scale_augmentation_severity = 0

        # confirm that severity is a float between [0,1]
        assert 0 <= jitter_augmentation_severity < 1
        assert 0 <= scale_augmentation_severity < 1

        self.rotation_flag = rotation_flag
        self.reflection_flag = reflection_flag
        self.jitter_augmentation_severity = jitter_augmentation_severity
        self.scale_augmentation_severity = scale_augmentation_severity

    def apply(self, img, orientation=0, reflect_x=0, reflect_y=0, jitter_x=0, jitter_y=0, scale_x=1.0, scale_y=1.0, **params):

        # apply the affine transformation
        img = apply_affine_transformation(img, orientation, reflect_x, reflect_y, jitter_x, jitter_y, scale_x, scale_y)

        # img = np.asarray(img, dtype=np.float32)
        return img

    def apply_to_mask(self, img, orientation=0, reflect_x=0, reflect_y=0, jitter_x=0, jitter_y=0, scale_x=1.0, scale_y=1.0, **params):
        img = np.asarray(img)

        # apply the affine transformation
        img = apply_affine_transformation(img, orientation, reflect_x, reflect_y, jitter_x, jitter_y, scale_x, scale_y)

        img = np.round(img)
        return img

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError("Method apply_to_bbox is not implemented in class " + self.__class__.__name__)

    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError("Method apply_to_keypoint is not implemented in class " + self.__class__.__name__)

    def apply_to_bboxes(self, bboxes, **params):
        raise NotImplementedError("Method apply_to_bboxes is not implemented in class " + self.__class__.__name__)

    def apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError("Method apply_to_keypoints is not implemented in class " + self.__class__.__name__)

    def get_params(self):
        # set default augmentation parameter values (which correspond to no transformation)
        orientation = 0
        reflect_x = False
        reflect_y = False
        jitter_x = 0
        jitter_y = 0
        scale_x = 1
        scale_y = 1

        if self.rotation_flag:
            orientation = 360 * np.random.rand()
        if self.reflection_flag:
            reflect_x = np.random.rand() > 0.5  # Bernoulli
            reflect_y = np.random.rand() > 0.5  # Bernoulli

        if self.jitter_augmentation_severity > 0:
            jitter_x = self.jitter_augmentation_severity * np.random.rand()
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_x = -1 * jitter_x
            jitter_y = self.jitter_augmentation_severity * np.random.rand()
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_y = -1 * jitter_y

        if self.scale_augmentation_severity > 0:
            max_val = 1 + self.scale_augmentation_severity
            min_val = 1 - self.scale_augmentation_severity
            scale_x = min_val + (max_val - min_val) * np.random.rand()
            scale_y = min_val + (max_val - min_val) * np.random.rand()


        return {
            "orientation": orientation,
            "reflect_x": reflect_x,
            "reflect_y": reflect_y,
            "jitter_x": jitter_x,
            "jitter_y": jitter_y,
            "scale_x": scale_x,
            "scale_y": scale_y,
        }

    def get_transform_init_args(self):
        return {
            "rotation_flag": self.rotation_flag,
            "reflection_flag": self.reflection_flag,
            "jitter_augmentation_severity": self.jitter_augmentation_severity,
            "scale_augmentation_severity": self.scale_augmentation_severity,
        }



