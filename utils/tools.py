import csv
import glob
import os
import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image
from ipywidgets import interact


def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


def jpg2npy(path):
    img_names = os.listdir(path)
    img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
    for file in img_names:
        sourceDir1 = os.path.join(path, file)
        im1 = cv2.imread(sourceDir1)
        im2 = np.array(im1)
        file_name = os.path.join(path, file.split('.')[0] + '.npy')
        np.save(file_name, im2)
        os.remove(sourceDir1)
    print('Done!')


def png2npy(path):
    img_names = os.listdir(path)
    img_names = list(filter(lambda x: x.endswith('.png'), img_names))
    for file in img_names:
        sourceDir1 = os.path.join(path, file)
        im1 = cv2.imread(sourceDir1)
        im2 = np.array(im1)
        file_name = os.path.join(path, file.split('.')[0] + '.npy')
        np.save(file_name, im2)
        os.remove(sourceDir1)
    print('Done!')


def thumbnail_pic(path, reslution):
    a = glob.glob(path + r'/*.jpg')
    for name in a:
        im = Image.open(name)
        im = im.resize(reslution)
        im.save(name, 'JPEG')
    print('Done!')


def show_seg(img, save_path=None, title=None, margin=0.05, dpi=80, cmap="gray"):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    slicer = False
    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]
        # the number of components is 3 or 4 consider it an RGB image
        if c not in (3, 4):
            slicer = True
    elif nda.ndim == 4:
        c = nda.shape[-1]
        if c not in (3, 4):
            raise RuntimeError("Unable to show 3D-vector Image")
        # take a z-slice
        slicer = True

    if slicer:
        ysize = nda.shape[1]
        xsize = nda.shape[2]
    else:
        ysize = nda.shape[0]
        xsize = nda.shape[1]

    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    def callback(z=None):
        extent = (0, xsize * spacing[1], ysize * spacing[0], 0)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        # Make the axis the right size...
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
        if z is None:
            image = nda
        else:
            image = nda[z, ...]
        ax.imshow(image, extent=extent, interpolation=None, cmap=cmap)
        if title:
            plt.title(title)
        if save_path is not None:
            plt.imsave(save_path, image)
        plt.show()

    if slicer:
        interact(callback, z=(0, nda.shape[0] - 1))
    else:
        callback()


def show_seg3d(img, save_path=None, xslices=[], yslices=[], zslices=[], title=None, margin=0.05, dpi=80):
    img_xslices = [img[s, :, :] for s in xslices]
    img_yslices = [img[:, s, :] for s in yslices]
    img_zslices = [img[:, :, s] for s in zslices]
    maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))
    img_null = sitk.Image([0, 0], img.GetPixelID(), img.GetNumberOfComponentsPerPixel())
    img_slices = []
    d = 0
    if len(img_xslices):
        img_slices += img_xslices + [img_null] * (maxlen - len(img_xslices))
        d += 1
    if len(img_yslices):
        img_slices += img_yslices + [img_null] * (maxlen - len(img_yslices))
        d += 1
    if len(img_zslices):
        img_slices += img_zslices + [img_null] * (maxlen - len(img_zslices))
        d += 1
    if maxlen != 0:
        if img.GetNumberOfComponentsPerPixel() == 1:
            img = sitk.Tile(img_slices, [maxlen, d])
        # TODO check in code to get Tile Filter working with VectorImages
        else:
            img_comps = []
            for i in range(0, img.GetNumberOfComponentsPerPixel()):
                img_slices_c = [sitk.VectorIndexSelectionCast(s, i) for s in img_slices]
                img_comps.append(sitk.Tile(img_slices_c, [maxlen, d]))
            img = sitk.Compose(img_comps)

    show_seg(img, save_path, title, margin, dpi)
