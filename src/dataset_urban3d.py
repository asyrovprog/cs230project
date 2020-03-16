import numpy as np
from mrcnn import utils
import skimage.io
import src.dataset as ds
import src.image_tools as it
from mrcnn import visualize
import os, random

class Urban3dDataset(utils.Dataset):
    def __init__(self, imagetype = "rgb"):
        super().__init__()

        self.imagetype = imagetype.upper()
        assert(self.imagetype in {"RGB", "RGBD", "D", "D2G", "RGBDT" })

        self.image_size = 256
        self.dataset = ds.urban3d_dataset()
        #self.dataset = ds.urban3d_dataset(root_folder='C:\\Study\\CS230\\project\\dataset\\spacenet-dataset\\urban3d')
        self.skip_empty_images = True

    def load(self, subset = "train", dataset_dir = None):
        # we have just one class
        self.add_class("building", 1, "building")

        # check subset valid
        assert(subset in {"train", "dev", "test"})
        input_files = self.dataset.filesX(subset, "RGB")
        random.shuffle(input_files)

        files_added = 0

        for i, file in enumerate(input_files):
            # get target (ground truth) for this input, we will use it to skip this input file is there is nothing in it,
            # see below 'if'
            classes = [1]
            if self.skip_empty_images:
                _, classes = self.load_file_mask(self.image_size, self.image_size, self.dataset.fileY(file))
            # if image has not buildings we skip it. ideally this should be handled automatically, but per following
            # thread is causes runtime error: https://github.com/matterport/Mask_RCNN/issues/1630
            if np.max(classes) > 0:
                self.add_image("building", path = file, image_id = files_added, width = self.image_size, height = self.image_size)
                files_added += 1
        print(f"Loaded {files_added} files for {subset}. {len(input_files) - files_added} files skipped (empty). Empty files cause div 0 in mrcnn lib.")

        # should have some images
        assert(len(self.image_info) > 0)

    def get_image_id(self, filename):
        basename = os.path.basename(filename)
        for fid in self.image_ids:
            curr = os.path.basename(self.source_image_link(fid))
            if basename == curr:
                return fid
        return -1

    def load_image_depth(self, image_id):
        # loading depth buffer image
        image_name = self.image_info[image_id]['path']
        image_name = image_name.replace("_RGB_", "_DSM_")
        image = skimage.io.imread(image_name) * 255.0

        assert(image.shape == (self.image_size, self.image_size))
        image.shape = (self.image_size, self.image_size, 1)

        return image

    def load_image_d2g(self, image_id):
        # loading depth buffer image
        image_name = self.image_info[image_id]['path']
        image_name = image_name.replace("_RGB_", "_DSM_")
        image = skimage.io.imread(image_name) * 255.0

        assert(image.shape == (self.image_size, self.image_size))
        image.shape = (self.image_size, self.image_size, 1)

        image = np.concatenate((image, image, image), axis = 2)

        return image

    def load_image_rgbd(self, image_id):
        # loading depth buffer image
        image_rgb_filename = self.image_info[image_id]['path']
        image_d_filename = image_rgb_filename.replace("_RGB_", "_DSM_")
        image = skimage.io.imread(image_rgb_filename)
        image_d = skimage.io.imread(image_d_filename) * 255.0

        # image = image / 255.0
        image_d.shape = (self.image_size, self.image_size, 1)
        image = np.concatenate((image, image_d), axis = 2)
        assert(image.shape == (self.image_size, self.image_size, 4))

        return image

    def load_image_rgbdt(self, image_id):
        # loading depth buffer image
        image_rgb_filename = self.image_info[image_id]['path']
        image_d_filename = image_rgb_filename.replace("_RGB_", "_DSM_")
        image_t_filename = image_rgb_filename.replace("_RGB_", "_DTM_")
        image = skimage.io.imread(image_rgb_filename)
        image_d = skimage.io.imread(image_d_filename) * 255.0
        image_t = skimage.io.imread(image_t_filename) * 255.0

        image_d.shape = (self.image_size, self.image_size, 1)
        image_t.shape = (self.image_size, self.image_size, 1)
        image = np.concatenate((image, image_d, image_t), axis = 2)
        assert(image.shape == (self.image_size, self.image_size, 5))

        return image

    def load_image_rgb(self, image_id):
        return super().load_image(image_id)

    def load_image(self, image_id):
        if self.imagetype == "RGB":
            return super().load_image(image_id)

        elif self.imagetype == "D":
            return self.load_image_depth(image_id)

        elif self.imagetype == "D2G":
            return self.load_image_d2g(image_id)

        elif self.imagetype == "RGBD":
            return self.load_image_rgbd(image_id)

        elif self.imagetype == "RGBDT":
            return self.load_image_rgbdt(image_id)

    def load_binary_mask(self, image_id, value = 255):

        image_info = self.image_info[image_id]
        filename = self.dataset.fileY(image_info["path"])
        mapimg = skimage.io.imread(filename)
        mask = mapimg > 0
        mapimg[mask] = value

        return mapimg

    def load_mask(self, image_id):

        """Generate instance masks for an image.
               Returns:
                masks: A bool array of shape [height, width, instance count] with
                    one mask per instance.
                class_ids: a 1D array of class IDs of the instance masks.
                """

        image_info = self.image_info[image_id]
        if image_info["source"] != "building":
            return super(self.__class__, self).load_mask(image_id)

        return self.load_file_mask(image_info["height"], image_info["width"], self.dataset.fileY(image_info["path"]))

    def load_file_mask(self, height, width, filename):

        mapimg = skimage.io.imread(filename)
        building_ids = []

        max_building_id = np.max(mapimg)
        if max_building_id > 0:
            for i in range(1, max_building_id + 1):
                t = np.sum((mapimg == i).astype(int))
                if t > 0:
                    building_ids.append(i)

        instance_count = int(max(1, len(building_ids)))
        mask = np.zeros((height, width, instance_count), dtype=np.bool)

        if len(building_ids) == 0:
            return mask, np.zeros([1], dtype=np.int32)

        i = 0
        for bid in building_ids:
            mask[:, :, i] = (mapimg == bid)
            i += 1

        classes = np.ones([mask.shape[-1]], dtype=np.int32)
        return mask, classes

    def image_reference(self, image_id):
        # Return the path of the image
        info = self.image_info[image_id]
        if info["source"] == "building":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def create_urban3d_model(stype: str, img_type = "RGB"):
    assert(stype in {"dev", "train", "test"})
    d = Urban3dDataset(img_type)
    if stype == "train":
        d.skip_empty_images = False
    d.load(stype)
    d.prepare()
    return d

if __name__ == "__main__":

    dataset_train = Urban3dDataset()
    dataset_train.load("train")
    dataset_train.prepare()

    images = [
        "TAM_Tile_108_RGB_0_6.tif"
    ]
    for image in images:
        image_id = dataset_train.get_image_id(image)
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)