import TransformationFile as Pillow


class StaticTransform:
    def __init__(self, type, order):
        self.type = type
        self.order = order


STATIC_TRANSFORMS = [
    StaticTransform('REFLECT_HORIZONTAL', 1),
    StaticTransform('REFLECT_VERTICAL', 1),
    StaticTransform('ROTATE_90', 2),
    StaticTransform('ROTATE_180', 2),
    StaticTransform('ROTATE_270', 2),
    StaticTransform('ROTATE_45', 3),
    StaticTransform('ROTATE_135', 3),
    StaticTransform('ROTATE_225', 3),
    StaticTransform('ROTATE_315', 3)
]


# Applies the static transform to the imgage, returning the resultant image
def apply_static_transform(im, static_transform):
    if static_transform.type == 'REFLECT_HORIZONTAL':
        return Pillow.reflect_horizontal(im)
    elif static_transform.type == 'REFLECT_VERTICAL':
        return Pillow.reflect_vertical(im)
    elif static_transform.type == 'ROTATE_90':
        return Pillow.rotate_90(im)
    elif static_transform.type == 'ROTATE_180':
        return Pillow.rotate_180(im)
    elif static_transform.type == 'ROTATE_270':
        return Pillow.rotate_270(im)
    elif static_transform.type == 'ROTATE_45':
        return Pillow.rotate_45(im)
    elif static_transform.type == 'ROTATE_135':
        return Pillow.rotate_135(im)
    elif static_transform.type == 'ROTATE_225':
        return Pillow.rotate_225(im)
    elif static_transform.type == 'ROTATE_315':
        return Pillow.rotate_315(im)
    else:
        print('Unrecognized transform:',
              static_transform.type, 'Image unchanged')
        return im


class Transform:
    def __init__(self, start_image=None):
        # The transforms that have been applied - can be anything from STATIC_TRANSFORMS
        self.static_transforms = []
        # The current image given the applied transforms listed in static_transoforms
        self.current_image = start_image
        self.add_image = None  # Image of what was added
        self.add_percent = None  # Measure of how much was added
        self.subtract_image = None  # Image of what was subtracted
        self.subtract_percent = None  # Measure of how much was subtracted
        self.score = None  # Just a generic score used to ranking Transforms

    # Alters the Transform by the static_transform provided
    def add_static_transform(self, static_transform):
        self.static_transforms.append(static_transform)
        self.current_image = apply_static_transform(
            self.current_image, static_transform)
        return self

    # Sets the addition info needed to reach the image provided
    def set_additions(self, im):
        self.add_image = Pillow.get_additions_image(self.current_image, im)
        self.add_percent = Pillow.percent(self.add_image)

    # Sets the subtraction info needed to reach the image provided
    def set_subtractions(self, im):
        self.subtract_image = Pillow.get_subtractions_image(
            self.current_image, im)
        self.subtract_percent = Pillow.percent(self.subtract_image)

    # Applies all the current transformations in this Transform to the provided image
    # Returns the resultant image after all transformations
    def apply_to(self, im):
        # Apply static transformations, in order
        for stat_trans in self.static_transforms:
            im = apply_static_transform(im, stat_trans)

        # Apply additions and subtractions
        if self.add_image is not None:
            im = Pillow.add_to(im, self.add_image)
        if self.subtract_image is not None:
            im = Pillow.subtract_from(im, self.subtract_image)

        return im
