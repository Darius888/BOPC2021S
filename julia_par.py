#! /usr/bin/env python

import numpy as np
import argparse
import time
from multiprocessing import Pool, TimeoutError
from math import sqrt, floor


def compute_julia_set_sequential(xmin, xmax, ymin, ymax, im_width, im_height):

    zabs_max = 10
    c = complex(-0.1, 0.65)
    nit_max = 1000

    xwidth  = xmax - xmin
    yheight = ymax - ymin

    julia = np.zeros((im_width, im_height))
    for ix in range(im_width):
        for iy in range(im_height):
            nit = 0
            # Map pixel position to a point in the complex plane
            z = complex(ix / im_width * xwidth + xmin,
                        iy / im_height * yheight + ymin)
            # Do the iterations
            while abs(z) <= zabs_max and nit < nit_max:
                z = z**2 + c
                nit += 1
            ratio = nit / nit_max
            julia[ix,iy] = ratio

    return julia

# Computes a patch of size patch_width, patch_height and returns the result, as well as the
# final x and y position of the computed patch.
def compute_julia_set_parallel(row, col, patch, patch_width, patch_height, x_min, y_min, x_max, y_max, im_width, im_height): # x_offset, y_offset,
    zabs_max = 10
    c = complex(-0.1, 0.65)
    nit_max = 1000

    x_width  = x_max - x_min
    y_height = y_max - y_min

    # We round down, at risk of losing some information at the border
    patch_width = floor(patch_width)
    patch_height = floor(patch_height)

    # The result size will be width x height
    julia = np.zeros((patch_width, patch_height))

    # Compute the image position of the patch
    start_x = col * patch
    end_x = start_x + patch_width
    start_y = row * patch
    end_y = start_y + patch_height

    # Loop through all patch positions
    for x_count, im_x in enumerate(range(start_x, end_x)):
        for y_count, im_y in enumerate(range(start_y, end_y)):
            nit = 0
            # Map pixel position to a point in the complex plane
            z = complex(im_x / im_width * x_width + x_min,
                        im_y / im_height * y_height + y_min)
            # Do the iterations
            while abs(z) <= zabs_max and nit < nit_max:
                z = z**2 + c
                nit += 1
            ratio = nit / nit_max
            julia[x_count,y_count] = ratio

    return start_x, end_x, start_y, end_y, julia


# Separates the images into patches and sends them to a Pool,
# then accumulates the results
def compute_julia_in_parallel(size, x_min, x_max, y_min, y_max, patch, nprocs):

    # -------- PARALLEL STRATEGY: -----------
    # separate image into patches, compute the julia set of each patch separately
    # then accumulate the results
    # most patches will be square, but some will have another width/height

    # ------- ERROR CHECKING ----------
    if patch > size:
        raise ValueError(f"Patch size {patch} larger than image size {size}.")

    num_patches = size / patch
    num_full_patches = floor(num_patches)

    task_list = []

    # -------- SQUARE PATCHES -----------
    for col in range(num_full_patches):
        for row in range(num_full_patches):
            # append task to task list
            task_list.append((row, col, patch, patch, patch, x_min, y_min, x_max, y_max, size, size)) # x_offset, y_offset,

    #  ----- LAST PATCHES --------
    # do the last patches as special case

    # 1. append the bottom right corner patch
    truncated_patch_size = (num_patches - num_full_patches) * patch
    task_list.append((num_full_patches, num_full_patches, patch, truncated_patch_size, truncated_patch_size, x_min, y_min, x_max, y_max, size, size)) # x_offset, y_offset,

    # 2. append the bottom patches
    for col in range(num_full_patches): # the part patches at the bottom
        # append task to task list
        task_list.append((num_full_patches, col, patch, patch, truncated_patch_size, x_min, y_min, x_max, y_max, size, size)) # x_offset, y_offset,

    # 3. append the rightmost patches
    for row in range(num_full_patches): # the part patches at the right
        # append task to task list
        task_list.append((row, num_full_patches, patch, truncated_patch_size, patch, x_min, y_min, x_max, y_max, size, size)) # x_offset, y_offset,
        #y_offset += y_step

    #  --------- RUN THE TASKS ----------
    with Pool(nprocs) as pool:
        results = pool.starmap(compute_julia_set_parallel, task_list)

    # ---------- ACCUMULATE THE RESULTS ------------
    full_image = np.zeros((size, size))
    for start_x, end_x, start_y, end_y, julia in results:
        # copy results into full_image
        full_image[start_x:end_x, start_y:end_y] = julia

    return full_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", help="image size in pixels (square images)", type=int, default=500)
    parser.add_argument("--xmin", help="", type=float, default=-1.5)
    parser.add_argument("--xmax", help="", type=float, default=1.5)
    parser.add_argument("--ymin", help="", type=float, default=-1.5)
    parser.add_argument("--ymax", help="", type=float, default=1.5)
    parser.add_argument("--patch", help="patch size in pixels (square images)", type=int, default=20)
    parser.add_argument("--nprocs", help="number of workers", type=int, default=1)
    parser.add_argument("-o", help="output file")
    args = parser.parse_args()

    #print(args)

    stime = time.perf_counter()
    julia_img = compute_julia_in_parallel(
        args.size,
        args.xmin, args.xmax,
        args.ymin, args.ymax,
        args.patch,
        args.nprocs)
    rtime = time.perf_counter() - stime

    print(f"{args.size};{args.patch};{args.nprocs};{rtime}")

    if not args.o is None:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        fig, ax = plt.subplots()
        ax.imshow(julia_img, interpolation='nearest', cmap=plt.get_cmap("hot"))
        plt.savefig(args.o)
        plt.show()
