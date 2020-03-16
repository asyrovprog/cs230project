import os, glob
from pathlib import Path
from typing import *
import skimage, skimage.io
import numpy as np

HOME_PATH = str(Path.home())

class urban3d_dataset:
    def __init__(self, root_folder: AnyStr = None):
        if root_folder is None:
            root_folder = os.getenv("DATASET_ROOT", f"{HOME_PATH}/data")
            root_folder = os.path.join(root_folder, "spacenet-dataset", "urban3d")
        self.folder = root_folder

    def filesX(self, dstype: str, filetype: str) -> List[str]:
        """
        Returns list of files from dataset for the specified parameters:
            dstype: 'train', 'dev' or 'test'
            filetype: 'RGB' or 'GTI'
        """
        dstype, filetype = dstype.lower(), filetype.upper()

        assert(dstype in {"train", "dev", "test"})
        assert(filetype in {"RGB", "GTI"}) # TODO: add more file types

        folder = os.path.join(self.folder, dstype, "inputs" if filetype != "GTI" else "target")
        assert(os.path.isdir(folder)) # folder must exists (build_dataset.py to create)

        pattern = os.path.join(folder, f"*_{filetype}_*.tif")
        assert(len(glob.glob(pattern)) > 0) # folder must have files (build_dataset.py to populate or check if path is correct)

        return glob.glob(pattern)

    def fileY(self, xfile: str) -> str:
        """
        Returns target file (building mask) for passed input file. Input file should be RGB or DSM or DTM
            xfile: input file, such as '.../inputs/JAX_Tile_004_RGB_0_2.tif'
        """
        base = os.path.basename(xfile)
        path = os.path.normpath(os.path.dirname(xfile))
        path = list(path.split(os.sep))

        assert(path[-1] == "inputs") # file must be from input folder

        for t in ["DTM", "RGB", "DTM"]:
            base = base.replace(t, "GTI")

        ret = os.path.join(self.folder,path[-2], "target", base)

        assert(os.path.isfile(ret)) # target file must exist
        return ret

    def mask_class(self, yfile_path, height, width):
        mapimg = skimage.io.imread(self.fileY(yfile_path))
        building_ids = []

        # create array of building ids (here we search for buiding ids and make sure
        # each is present in image)
        max_building_id = np.max(mapimg)
        if max_building_id > 0:
            for i in range(1, max_building_id):
                t = np.sum((mapimg == i).astype(int))
                if t > 0:
                    building_ids.append(i)

        # we create a slice in array for each building, so if we our image is 256x256 and we have 16
        # building then result shape is (256,256,16). If now buildings then we have just one slice filled with 0
        instance_count = int(max(1, len(building_ids)))
        mask = np.zeros((height, width, instance_count), dtype=np.bool)

        # exit with empty mask if now buildings in this image
        if len(building_ids) == 0:
            return mask, np.zeros([1], dtype=np.int32)

        # put mask of each building in it's own slice
        for i, bid in enumerate(building_ids):
            mask[:, :, i] = (mapimg == bid)

        classes = np.ones([mask.shape[-1]], dtype=np.int32)
        return mask, classes


if __name__ == "__main__":

    ds = urban3d_dataset()

    for dt in ["train", "dev", "test"]:
        for f in ds.filesX(dt, "RGB"):
            print(f, ds.fileY(f))
