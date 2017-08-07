import cv2
import numpy as np

def white_to_alpha(img, color=None):
    """ Convert white values in an image to alpha.  """

    if color is None:
        color = [0, 0, 0]

    img[:, :, 3] = 255 - img[:, :, 0]

    img[:, :, 0:3] = color

    img_clipped = np.minimum(255 - img[:, :, 3], 40)
    np.putmask(img[:, :, 3], img[:, :, 3] > 30, img[:, :, 3] + img_clipped)

def add_alpha_channel(img):
    b, g, r = cv2.split(img)

    a = np.ones(b.shape, np.uint8) * 255

    return cv2.merge((b, g, r, a))

def alpha_composite(face_img, overlay_t_img):
    """ From https://stackoverflow.com/a/37198079 """
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

