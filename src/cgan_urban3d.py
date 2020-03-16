from src.cgan import *
import random as rn
from src.mrcnn_imports import *
from configs.config_factory import *
from src.training import *
from src.image_tools import *
from keras.preprocessing.image import *
from itertools import *
import random
from skimage import io
import numpy as np

class Urban3dCondGANConfig(CondGANConfig):
    NAME = "urban3d_cond_gan_rgbd"

    COND_SHAPE     = (256, 256, 4) # 4 channel are for RGBD, etc
    GEN_SHAPE      = (256, 256, 1) # 1 channel for building footprint

    IMAGE_FOLDER   = os.path.join("logs", NAME, "images")
    MODEL_FOLDER   = os.path.join("logs", NAME, "models")

    def __init__(self, train = "train", val = "dev", itype = None):

        assert(train is None or train == "train")
        assert(val in {"dev", "test", "train"})

        if itype is None:
            channels = self.COND_SHAPE[2]
            if channels == 1:
                itype = "D"
            elif channels == 3:
                itype = "RGB"
            elif channels == 4:
                itype = "RGBD"
            elif channels == 5:
                itype = "RGBDT"
            else:
                raise Exception("Unsupported channels")

        if not train is None:
            self.ds = ds.create_urban3d_model(train, itype)

        self.ds_val = ds.create_urban3d_model(val, itype)
        self.image_gens = self.create_image_generators()

    def num_samples(self, is_train = True):
        dst = self.ds if is_train else self.ds_val
        return len(dst.image_info)

    def batch(self, bid, batch_size, is_train = True):
        """
        Return batch of specified size from training or evaluation dataset starting batch id 'bid'
        """
        dst        = self.ds if is_train else self.ds_val
        image_info = dst.image_info
        ids        = [i for i in range(bid * batch_size, min(len(image_info), (bid + 1) * batch_size))]

        if len(ids) < batch_size:
            ids = ids + [rn.randint(0, len(image_info) - 1) for _ in range(batch_size - len(ids))]

        return self.batch_from_ids(ids, dst)


    def random_batch(self, batch_size, is_train = True):
        """
        Return random batch of specified size from training or evaluation dataset
        """
        dst = self.ds if is_train else self.ds_val
        image_info = dst.image_info

        ids = [rn.randint(0, len(image_info) - 1) for _ in range(batch_size)]

        return self.batch_from_ids(ids, dst)

    def batch_from_ids(self, ids, dst, use_image_generator = False, batch_size = None):
        """
        Return random batch of specified size from training or evaluation dataset
            ids: list of image ids
            dst: dataset (either self.ds or self.ds_val) that that is source of 'ids' images
        """
        image_info = dst.image_info

        imgs_target = []  # true building footprint
        imgs_sat = []     # rgb satellite image
        imgs_filenames = []
        channels = self.COND_SHAPE[2]

        def rescale(img):
            return (img.astype(np.float32) - 127.5) / 127.5

        for i in ids:
            image_id = image_info[i]["id"]
            imgs_filenames.append(image_info[i]["path"])

            if channels == 1:
                img = dst.load_image_depth(image_id)
            elif channels == 3:
                img = dst.load_image_rgb(image_id)
            elif channels == 4:
                img = dst.load_image_rgbd(image_id)
            elif channels == 5:
                img = dst.load_image_rgbdt(image_id)
            else:
                raise Exception("Unsupported channels")

            imgs_sat.append(img)

            mask = dst.load_binary_mask(image_id)
            mask.shape = (mask.shape[0], mask.shape[1], 1)

            imgs_target.append(mask)

        if use_image_generator:
            if batch_size is None:
                batch_size = len(ids)
            gens = self.init_image_generator(np.array(imgs_sat), np.array(imgs_target), self.image_gens, batch_size)
            img_target, img_sat = [], []
            for x, y in zip(gens[0], gens[1]):
                if len(img_target) >= batch_size:
                    break
                for i in range(x.shape[0]):
                    img_sat.append(x[i])
                    img_target.append(y[i])

        imgs_target, imgs_sat = rescale(np.array(imgs_target)), rescale(np.array(imgs_sat))
        return imgs_target, imgs_sat, imgs_filenames

    def create_image_generators(self):

        # we create two instances with the same arguments
        data_gen = dict(rotation_range = 45.,
                        horizontal_flip = True,
                        vertical_flip = True,
                        zoom_range = (0.7, 1),
                        shear_range = 1.4)

        image_datagen = ImageDataGenerator(**data_gen)
        mask_datagen = ImageDataGenerator(**data_gen)

        return image_datagen, mask_datagen

    def init_image_generator(self, img_x, img_y, genpair, batch_size):
        seed = random.randint(0, int(1e8))

        image_gen_x: ImageDataGenerator = genpair[0]
        image_gen_y: ImageDataGenerator = genpair[1]

        # Provide the same seed and keyword arguments to the fit and flow methods seed = 1
        image_gen_x.fit(img_x, augment = True, seed = seed)
        image_gen_y.fit(img_y, augment = True, seed = seed)

        image_gen_x = image_gen_x.flow(x = img_x, batch_size = batch_size, seed = seed)
        image_gen_y = image_gen_y.flow(x = img_y, batch_size = batch_size, seed = seed)

        return image_gen_x, image_gen_y