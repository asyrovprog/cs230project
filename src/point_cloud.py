import argparse
import skimage.io
import math
import sys, os
import numpy as np
import open3d as o3d
from collections import *
import random

IMAGE_ROOT = "analysis/point_cloud"
EPSILON = sys.float_info.epsilon
MASK_OFFSET = 0.05

def create_pointcloud_ext(rgb, depth, masks, mask_colors, ply_file):

    SIZE = rgb.shape[0]

    if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
        raise Exception("Mismatch in RGB and Depth image shapes")

    centerX = rgb.shape[1] / 2.0
    centerY = rgb.shape[0] / 2.0
    scaleZ  = rgb.shape[1] / 5

    points = []
    height = defaultdict(float)

    for v in range(rgb.shape[1]):
        for u in range(rgb.shape[0]):
            color = rgb[u][v]
            Z = depth[u][v]

            if Z < EPSILON:
                continue

            Z *= scaleZ
            X = int(u - centerX)
            Y = int(v - centerY)
            height[(X, Y)] = Z

            points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))

    def is_border(u, v, mask):
        if u == 0 or v == 0 or u == SIZE - 1 or v == SIZE - 1:
            return False
        val = mask[u, v]
        if val > 0:
            if (mask[u, v - 1] != val) or (mask[u, v + 1] != val) or (mask[u - 1, v] != val) or (mask[u + 1, v] != val):
                return True
        return False

    i = 0
    for mask, clr in zip(masks, mask_colors):
        for v in range(mask.shape[1]):
            for u in range(mask.shape[0]):
                X = int(u - centerX)
                Y = int(v - centerY)
                if (X, Y) in height:
                    if np.random.uniform() < 1.0 / len(masks):
                        if is_border(u, v, mask):
                            Z = height[(X, Y)] + ((i + 1) * MASK_OFFSET * scaleZ)
                            points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, clr[0], clr[1], clr[2]))
        i += 1


    file = open(ply_file, "w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
''' % (len(points), "".join(points)))
    file.close()

def create_pointcloud(rgb_file, depth_file, mask_file, ply_file):
    rgb = skimage.io.imread(rgb_file)
    depth = skimage.io.imread(depth_file)
    mask = skimage.io.imread(mask_file)

    create_pointcloud_ext(rgb, depth, [mask], [(255, 0, 0)], ply_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate PLY format 3d point cloud based on rgb and depth buffer images.')
    parser.add_argument('--rgb_file', default = "rgb.tif", help='Input rgb file')
    parser.add_argument('--depth_file', default = "dsm.tif", help='Input depth buffer file')
    parser.add_argument('--mask_file', default="mask.tif", help='Image GT mask file')
    parser.add_argument('--ply_file', default = "img.ply", help='Output PLY file')
    parser.add_argument('--visualize', default = True, help = 'Open visualization screen with created PLY file')
    args = parser.parse_args()

    # create point cloud from rgb images
    create_pointcloud(os.path.join(IMAGE_ROOT, args.rgb_file),
                      os.path.join(IMAGE_ROOT, args.depth_file),
                      os.path.join(IMAGE_ROOT, args.mask_file),
                      os.path.join(IMAGE_ROOT, args.ply_file))

    if args.visualize:
        # Read the point cloud
        pcd = o3d.io.read_point_cloud(os.path.join(IMAGE_ROOT, args.ply_file))

        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])

