import albumentations
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F
import pandas as pd
import numpy as np

class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.
    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)
    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / 2),
                             int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value

                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')

def add_mask(xs,p,gd,height=300,width=300,resize=(lambda x:x)):
    '''
    add mask based on the probability p
    '''
    r = np.random.random()
    if r > p:
        return xs
    m = xs.shape[0]
    rotate = np.random.randint(0,21)
    gd.init_masks(height,width)
    for i in range(m):
        to_add = xs[i,:,:,:]
        xs[i,:,:,:]= gd.apply(to_add,gd.masks[0],0,0,angle=rotate)
    return xs

def onehot(ys,nums=[168,11,7]):
    '''
    single y to onehot y
    nums refer to the class
    '''
    ress=[]
    batch = ys[0].shape[0]
    for y,num in zip(ys,nums):
        res = np.zeros((y.shape[0],num))
        for i in range(batch):
            res[i,int(y[i])] = 1
        ress.append(res)
    return ress

def val_generator(generator):
    '''
    make generator of validation set
    '''
    while True:
        x,ys = next(generator)
        yield x,onehot(ys)

def train_generator(generator,f,steps_per_epoch=8):
    '''
    make generator of training set
    '''
    i = 0
    p=0.25
    divide = 0
    while True:
        i += steps_per_epoch
        grids = np.random.randint(4,8)
        gd = GridMask(num_grid=grids,p=1,rotate=30)
        xs,y = next(generator)

        xs = f(xs,p,gd,300,300)
        if p<0.85 and int(i / 180074) > divide:
            p += 0.15
            divide+=1
        yield xs,onehot(y)
