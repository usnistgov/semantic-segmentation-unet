import numpy as np
import scipy
import scipy.ndimage
import scipy.signal
import skimage.io
import skimage.transform


# Boxes are [N, 5] with the columns being [x, y, w, h, class-id]
def augment_image_box_pair(img, boxes, rotation_flag=False, reflection_flag=False,
                           jitter_augmentation_severity=0,  # jitter augmentation severity as a fraction of the image size
                           noise_augmentation_severity=0,  # noise augmentation as a percentage of current noise
                           scale_augmentation_severity=0,  # scale augmentation as a percentage of the image size):
                           blur_augmentation_max_sigma=0,  # blur augmentation kernel maximum size):
                           intensity_augmentation_severity=0):  # intensity augmentation as a percentage of the current intensity

    assert rotation_flag == False, "Rotation not implemented for image and boxes pair"
    img = np.asarray(img)

    # ensure input images are np arrays
    img = np.asarray(img, dtype=np.float32)

    debug_worst_possible_transformation = False # useful for debuging how bad images can get

    # check that the input image and mask are 2D images
    assert len(img.shape) == 2

    # convert input Nones to expected
    if jitter_augmentation_severity is None:
        jitter_augmentation_severity = 0
    if noise_augmentation_severity is None:
        noise_augmentation_severity = 0
    if scale_augmentation_severity is None:
        scale_augmentation_severity = 0
    if blur_augmentation_max_sigma is None:
        blur_augmentation_max_sigma = 0
    if intensity_augmentation_severity is None:
        intensity_augmentation_severity = 0

    # confirm that severity is a float between [0,1]
    assert 0 <= jitter_augmentation_severity < 1
    assert 0 <= noise_augmentation_severity < 1
    assert 0 <= scale_augmentation_severity < 1
    assert 0 <= intensity_augmentation_severity < 1

    # get the size of the input image
    h, w = img.shape

    # set default augmentation parameter values (which correspond to no transformation)
    reflect_x = False
    reflect_y = False
    jitter_x = 0
    jitter_y = 0
    scale_x = 1
    scale_y = 1

    if reflection_flag:
        reflect_x = np.random.rand() > 0.5  # Bernoulli
        reflect_y = np.random.rand() > 0.5  # Bernoulli
    if jitter_augmentation_severity > 0:
        if debug_worst_possible_transformation:
            jitter_x = int(jitter_augmentation_severity * (w * 1))  # uniform random jitter integer
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_x = -1 * jitter_x

            jitter_y = int(jitter_augmentation_severity * (h * 1))  # uniform random jitter integer
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_y = -1 * jitter_y
        else:
            jitter_x = int(jitter_augmentation_severity * (w * np.random.rand()))  # uniform random jitter integer
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_x = -1 * jitter_x

            jitter_y = int(jitter_augmentation_severity * (h * np.random.rand()))  # uniform random jitter integer
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_y = -1 * jitter_y

    if scale_augmentation_severity > 0:
        max_val = 1 + scale_augmentation_severity
        min_val = 1 - scale_augmentation_severity
        if debug_worst_possible_transformation:
            scale_x = min_val + (max_val - min_val) * 1
            scale_y = min_val + (max_val - min_val) * 1
        else:
            scale_x = min_val + (max_val - min_val) * np.random.rand()
            scale_y = min_val + (max_val - min_val) * np.random.rand()


    # apply the affine transformation
    img = apply_affine_transformation(img, 0, reflect_x, reflect_y, jitter_x, jitter_y, scale_x, scale_y)

    # apply the transformation elements to the boxes
    boxes = apply_affine_transformation_boxes(boxes, img.shape, reflect_x, reflect_y, jitter_x, jitter_y, scale_x, scale_y)

    # apply augmentations
    if noise_augmentation_severity > 0:
        sigma_max = noise_augmentation_severity * (np.max(img) - np.min(img))
        max_val = sigma_max
        min_val = -sigma_max
        if debug_worst_possible_transformation:
            sigma = min_val + (max_val - min_val) * 1
        else:
            sigma = min_val + (max_val - min_val) * np.random.rand()
        sigma_img = np.random.randn(img.shape[0], img.shape[1]) * sigma
        img = img + sigma_img

    # apply blur augmentation
    if blur_augmentation_max_sigma > 0:
        max_val = blur_augmentation_max_sigma
        min_val = -blur_augmentation_max_sigma
        if debug_worst_possible_transformation:
            sigma = min_val + (max_val - min_val) * 1
        else:
            sigma = min_val + (max_val - min_val) * np.random.rand()
        if sigma < 0:
            sigma = 0
        if sigma > 0:
            img = scipy.ndimage.filters.gaussian_filter(img, sigma, mode='reflect')

    if intensity_augmentation_severity > 0:
        img_range = np.max(img) - np.min(img)
        if debug_worst_possible_transformation:
            value = 1 * intensity_augmentation_severity * img_range
        else:
            value = np.random.rand() * intensity_augmentation_severity * img_range
        if np.random.rand() > 0.5:
            sign = 1.0
        else:
            sign = -1.0
        delta = sign * value
        img = img + delta # additive intensity adjustment

    img = np.asarray(img, dtype=np.float32)
    return img, boxes


def augment_image(img, mask=None, rotation_flag=False, reflection_flag=False,
                  jitter_augmentation_severity=0,  # jitter augmentation severity as a fraction of the image size
                  noise_augmentation_severity=0,  # noise augmentation as a percentage of current noise
                  scale_augmentation_severity=0,  # scale augmentation as a percentage of the image size):
                  blur_augmentation_max_sigma=0,  # blur augmentation kernel maximum size):
                  intensity_augmentation_severity=0):  # intensity augmentation as a percentage of the current intensity

    img = np.asarray(img)

    # ensure input images are np arrays
    img = np.asarray(img, dtype=np.float32)

    debug_worst_possible_transformation = False # useful for debuging how bad images can get

    # check that the input image and mask are 2D images
    assert len(img.shape) == 2

    # convert input Nones to expected
    if jitter_augmentation_severity is None:
        jitter_augmentation_severity = 0
    if noise_augmentation_severity is None:
        noise_augmentation_severity = 0
    if scale_augmentation_severity is None:
        scale_augmentation_severity = 0
    if blur_augmentation_max_sigma is None:
        blur_augmentation_max_sigma = 0
    if intensity_augmentation_severity is None:
        intensity_augmentation_severity = 0

    # confirm that severity is a float between [0,1]
    assert 0 <= jitter_augmentation_severity < 1
    assert 0 <= noise_augmentation_severity < 1
    assert 0 <= scale_augmentation_severity < 1
    assert 0 <= intensity_augmentation_severity < 1

    # get the size of the input image
    h, w = img.shape

    if mask is not None:
        mask = np.asarray(mask, dtype=np.float32)
        assert len(mask.shape) == 2
        assert (mask.shape[0] == h and mask.shape[1] == w)

    # set default augmentation parameter values (which correspond to no transformation)
    orientation = 0
    reflect_x = False
    reflect_y = False
    jitter_x = 0
    jitter_y = 0
    scale_x = 1
    scale_y = 1

    if rotation_flag:
        orientation = 360 * np.random.rand()
    if reflection_flag:
        reflect_x = np.random.rand() > 0.5  # Bernoulli
        reflect_y = np.random.rand() > 0.5  # Bernoulli
    if jitter_augmentation_severity > 0:
        if debug_worst_possible_transformation:
            jitter_x = int(jitter_augmentation_severity * (w * 1))  # uniform random jitter integer
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_x = -1 * jitter_x

            jitter_y = int(jitter_augmentation_severity * (h * 1))  # uniform random jitter integer
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_y = -1 * jitter_y
        else:
            jitter_x = int(jitter_augmentation_severity * (w * np.random.rand()))  # uniform random jitter integer
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_x = -1 * jitter_x

            jitter_y = int(jitter_augmentation_severity * (h * np.random.rand()))  # uniform random jitter integer
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_y = -1 * jitter_y

    if scale_augmentation_severity > 0:
        max_val = 1 + scale_augmentation_severity
        min_val = 1 - scale_augmentation_severity
        if debug_worst_possible_transformation:
            scale_x = min_val + (max_val - min_val) * 1
            scale_y = min_val + (max_val - min_val) * 1
        else:
            scale_x = min_val + (max_val - min_val) * np.random.rand()
            scale_y = min_val + (max_val - min_val) * np.random.rand()

    # apply the affine transformation
    img = apply_affine_transformation(img, orientation, reflect_x, reflect_y, jitter_x, jitter_y, scale_x, scale_y)
    if mask is not None:
        mask = apply_affine_transformation(mask, orientation, reflect_x, reflect_y, jitter_x, jitter_y, scale_x, scale_y)

    # apply augmentations
    if noise_augmentation_severity > 0:
        sigma_max = noise_augmentation_severity * (np.max(img) - np.min(img))
        max_val = sigma_max
        min_val = -sigma_max
        if debug_worst_possible_transformation:
            sigma = min_val + (max_val - min_val) * 1
        else:
            sigma = min_val + (max_val - min_val) * np.random.rand()
        sigma_img = np.random.randn(img.shape[0], img.shape[1]) * sigma
        img = img + sigma_img

    # apply blur augmentation
    if blur_augmentation_max_sigma > 0:
        max_val = blur_augmentation_max_sigma
        min_val = -blur_augmentation_max_sigma
        if debug_worst_possible_transformation:
            sigma = min_val + (max_val - min_val) * 1
        else:
            sigma = min_val + (max_val - min_val) * np.random.rand()
        if sigma < 0:
            sigma = 0
        if sigma > 0:
            img = scipy.ndimage.filters.gaussian_filter(img, sigma, mode='reflect')

    if intensity_augmentation_severity > 0:
        img_range = np.max(img) - np.min(img)
        if debug_worst_possible_transformation:
            value = 1 * intensity_augmentation_severity * img_range
        else:
            value = np.random.rand() * intensity_augmentation_severity * img_range
        if np.random.rand() > 0.5:
            sign = 1.0
        else:
            sign = -1.0
        delta = sign * value
        img = img + delta # additive intensity adjustment

    img = np.asarray(img, dtype=np.float32)
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float32)
        mask = np.round(mask)
        return img, mask
    else:
        return img


def apply_affine_transformation_boxes(boxes, input_shape, reflect_x, reflect_y, jitter_x, jitter_y, scale_x, scale_y):
    # Boxes are [N, 5] with the columns being [x, y, w, h, class-id]

    if boxes.shape[0] == 0: return

    # convert Boxes to [x_st, y_st, x_end, y_end]
    class_id = boxes[:, 4]
    x_st = boxes[:, 0]
    y_st = boxes[:, 1]
    x_end = boxes[:, 0] + boxes[:, 2] - 1
    y_end = boxes[:, 1] + boxes[:, 3] - 1

    x_st = x_st * scale_x + jitter_x
    x_end = x_end * scale_x + jitter_x
    y_st = y_st * scale_y + jitter_y
    y_end = y_end * scale_y + jitter_y

    h = input_shape[0]
    w = input_shape[1]

    # filter out boxes which no longer exist within the image
    idx = np.logical_or(np.logical_or(x_st >= w, y_st >= h), np.logical_or(x_end < 0, y_end < 0))
    if np.any(idx):
        idx = np.logical_not(idx)
        x_st = x_st[idx]
        y_st = y_st[idx]
        x_end = x_end[idx]
        y_end = y_end[idx]
        class_id = class_id[idx]

    # handle the case where we remove all boxes
    if len(x_st) == 0:
        return None

    # constrain to input shape
    x_st = np.maximum(x_st, 0)
    y_st = np.maximum(y_st, 0)

    x_end = np.minimum(x_end, w - 1)
    y_end = np.minimum(y_end, h - 1)

    # perform reflection
    if reflect_x:
        old_x_st = x_st
        old_x_end = x_end
        x_st = w - old_x_end
        x_end = w - old_x_st
    if reflect_y:
        old_y_st = y_st
        old_y_end = y_end
        y_st = h - old_y_end
        y_end = h - old_y_st

    # convert Boxes to [x, y, w, h]
    w = x_end - x_st + 1
    h = y_end - y_st + 1

    assert (np.all(h > 0) and np.all(w > 0)), 'box with zero or negative size'

    x_st = x_st.reshape(-1, 1)
    y_st = y_st.reshape(-1, 1)
    w = w.reshape(-1, 1)
    h = h.reshape(-1, 1)
    class_id = class_id.reshape(-1, 1)

    boxes = np.hstack((x_st, y_st, w, h, class_id)).astype(np.int32)
    return boxes


def apply_affine_transformation(I, orientation, reflect_x, reflect_y, jitter_x, jitter_y, scale_x, scale_y):

    if orientation is not 0:
        I = skimage.transform.rotate(I, orientation, preserve_range=True, mode='reflect')

    tform = skimage.transform.AffineTransform(translation=(jitter_x, jitter_y),
                                              scale=(scale_x, scale_y))
    I = skimage.transform.warp(I, tform._inv_matrix, mode='reflect', preserve_range=True)

    if reflect_x:
        I = np.fliplr(I)
    if reflect_y:
        I = np.flipud(I)

    return I
