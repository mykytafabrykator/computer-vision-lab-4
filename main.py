import os
import matplotlib.pyplot as plt
from skimage import (
    io, morphology, filters,
    measure, img_as_ubyte, img_as_float,
    color,
)
from skimage.morphology import disk, reconstruction
from skimage.filters import rank
import numpy as np

os.makedirs('result_images', exist_ok=True)


# 1. Дилатація
def dilation(image_path):
    image = io.imread(image_path, as_gray=True)
    dilated = morphology.dilation(image, disk(1.5))
    plt.imsave('result_images/task1.jpg', dilated, cmap='gray')


# 2. Ерозія
def erosion(image_path):
    image = io.imread(image_path, as_gray=True)
    eroded = morphology.erosion(image, disk(5))
    plt.imsave('result_images/task2.jpg', eroded, cmap='gray')


# 3. Розмикання та замикання
def opening_closing(image_path):
    image = io.imread(image_path, as_gray=True)
    opened = morphology.opening(image, disk(1.9))
    closed = morphology.closing(opened, disk(1.9))
    plt.imsave('result_images/task3_1.jpg', opened, cmap='gray')
    plt.imsave('result_images/task3_2.jpg', closed, cmap='gray')


# 4. Потоншення
def thinning(image_path):
    image = io.imread(image_path, as_gray=True)
    threshold = filters.threshold_otsu(image)
    binary_image = image > threshold
    cleaned_image = morphology.opening(binary_image, morphology.disk(1))
    thinned_image = morphology.thin(cleaned_image)
    plt.imsave('result_images/task4.jpg', thinned_image, cmap='gray')


# 5. Побудова остова
def skeleton(image_path):
    image = io.imread(image_path, as_gray=True)
    threshold = filters.threshold_otsu(image)
    binary_image = image > threshold
    cleaned_image = morphology.opening(binary_image, morphology.disk(1))
    skeleton = morphology.skeletonize(cleaned_image)
    plt.imsave('result_images/task5.jpg', skeleton, cmap='gray')


# 6. Виділення компонент зв’язності
def connected_components(image_path):
    image = io.imread(image_path, as_gray=True)

    threshold = filters.threshold_otsu(image)
    binary_image = image > threshold

    cleaned_image = morphology.remove_small_objects(binary_image, min_size=50)

    labeled_image = measure.label(cleaned_image)
    properties = measure.regionprops(labeled_image)

    labeled_rgb = color.label2rgb(labeled_image, bg_label=0)

    fig, ax = plt.subplots()
    ax.imshow(labeled_rgb)

    for prop in properties:
        y0, x0 = prop.centroid
        ax.plot(x0, y0, 'ro', markersize=5)
    ax.axis('off')
    fig.savefig('result_images/task6.jpg')


# 7. Морфологічна реконструкція
def morphological_reconstruction_14(mask_path, marker_path):
    marker = img_as_float(io.imread(marker_path, as_gray=True))
    mask = img_as_float(io.imread(mask_path, as_gray=True))

    marker = (marker > 0.5).astype(float)
    mask = (mask > 0.5).astype(float)

    marker = np.minimum(marker, mask)

    reconstructed = morphology.reconstruction(marker, mask, method='dilation')

    plt.imsave('result_images/task7.jpg', reconstructed, cmap='gray')


# 8. Морфологічна реконструкція (слайд 15)
def morphological_reconstruction_15(image_path):
    image = img_as_float(io.imread(image_path, as_gray=True))

    seed = np.minimum(morphology.erosion(image, disk(1)), image)

    reconstructed = reconstruction(seed, image, method='dilation')
    plt.imsave('result_images/task8.jpg', reconstructed, cmap='gray')


# 9. Півтонова дилатація та ерозія
def grayscale_dilation_erosion(image_path):
    image = io.imread(image_path, as_gray=True)
    dilated = rank.maximum(img_as_ubyte(image), disk(3))
    eroded = rank.minimum(img_as_ubyte(image), disk(3))
    plt.imsave('result_images/task9_1.jpg', dilated, cmap='gray')
    plt.imsave('result_images/task9_2.jpg', eroded, cmap='gray')


# 10. Півтонове розмикання та замикання
def grayscale_opening_closing(image_path):
    image = io.imread(image_path, as_gray=True)
    opened = morphology.opening(image, disk(3))
    closed = morphology.closing(image, disk(3))
    plt.imsave('result_images/task10_1.jpg', opened, cmap='gray')
    plt.imsave('result_images/task10_2.jpg', closed, cmap='gray')


# 11. Морфологічний градієнт
def morphological_gradient(image_path):
    image = io.imread(image_path, as_gray=True)
    gradient = (morphology.dilation(image, disk(1))
                - morphology.erosion(image, disk(1)))
    plt.imsave('result_images/task11.jpg', gradient, cmap='gray')


# 12. Перетворення «виступ» (Top-hat)
def tophat_transformation(image_path):
    image = img_as_ubyte(io.imread(image_path, as_gray=True))

    selem = disk(40)

    opened = morphology.opening(image, selem)
    top_hat = image - opened

    closed = morphology.closing(image, selem)
    bottom_hat = closed - image

    plt.imsave('result_images/task12_1.jpg', top_hat, cmap='gray')
    plt.imsave('result_images/task12_2.jpg', bottom_hat, cmap='gray')


# 13. Морфологічна півтонова реконструкція
def grayscale_morphological_reconstruction(image_path):
    image = img_as_ubyte(io.imread(image_path, as_gray=True))
    selem = disk(3)
    eroded = rank.minimum(image, selem)
    reconstructed = reconstruction(eroded, image, method='dilation')
    result = image - reconstructed
    plt.imsave('result_images/task13.jpg', result, cmap='gray')


if __name__ == "__main__":
    dilation("input/pic.1.jpg")
    erosion("input/pic.2.jpg")
    opening_closing("input/pic.3.jpg")
    thinning("input/pic.3.jpg")
    skeleton("input/pic.4.jpg")
    connected_components("input/pic.5.jpg")
    morphological_reconstruction_14("input/pic.6a.tif", "input/pic.6b.tif")
    morphological_reconstruction_15("input/pic.7.png")
    grayscale_dilation_erosion("input/pic.8.png")
    grayscale_opening_closing("input/pic.9.png")
    morphological_gradient("input/cameraman.tif")
    tophat_transformation("input/rice.png")
    grayscale_morphological_reconstruction("input/pic.10.png")
