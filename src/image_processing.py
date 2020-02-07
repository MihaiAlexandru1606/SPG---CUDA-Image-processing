from PIL import Image, ImageTk
from src.kernel_function import *
from src.global_var import *
from src.io_image import display_new_image
from copy import deepcopy

BLOCK_SIZE = 30


def update_image(function, name):
    if not GlobalVar.is_image:
        print("Nu este nici o imagine incarcata")
        return

    grid = (GlobalVar.width // BLOCK_SIZE + 1, GlobalVar.height // BLOCK_SIZE + 1, 1)
    function(grid, (BLOCK_SIZE, BLOCK_SIZE, 1), (GlobalVar.pixels, GlobalVar.width, GlobalVar.height))
    GlobalVar.name_operation = name

    display_new_image()


def update_image2(function, name):
    if not GlobalVar.is_image:
        print("Nu este nici o imagine incarcata")
        return

    grid = (GlobalVar.width // BLOCK_SIZE + 1, GlobalVar.height // BLOCK_SIZE + 1, 1)
    input_pixel = deepcopy(GlobalVar.pixels)
    function(grid, (BLOCK_SIZE, BLOCK_SIZE, 1), (input_pixel, GlobalVar.pixels, GlobalVar.width, GlobalVar.height))
    GlobalVar.name_operation = name

    display_new_image()


def update_image4(function, name):
    if not GlobalVar.is_image:
        print("Nu este nici o imagine incarcata")
        return

    grid = (GlobalVar.width // BLOCK_SIZE + 1, GlobalVar.height // BLOCK_SIZE + 1, 1)
    input_pixel = deepcopy(GlobalVar.pixels)
    function(grid, (BLOCK_SIZE, BLOCK_SIZE, 1), (input_pixel, GlobalVar.pixels, GlobalVar.width, GlobalVar.height))
    GlobalVar.name_operation = name


def update_image3(function, name, param):
    if not GlobalVar.is_image:
        print("Nu este nici o imagine incarcata")
        return

    grid = (GlobalVar.width // BLOCK_SIZE + 1, GlobalVar.height // BLOCK_SIZE + 1, 1)
    function(grid, (BLOCK_SIZE, BLOCK_SIZE, 1), (GlobalVar.pixels, GlobalVar.width, GlobalVar.height, param))
    GlobalVar.name_operation = name

    display_new_image()


def update_grayscale():
    update_image(gray_scale, "grayscale")
    print("Gata : Gray Scale")


def update_red_only():
    update_image(red, "red_only")
    print("Gata : Red Only")


def update_blue_only():
    update_image(blue, "blue_only")
    print("Gata : Blue Only")


def update_green_only():
    update_image(green, "green_only")
    print("Gata: Green Only")


def update_inverse():
    update_image(inverse, "inverse")
    print("Gata: Inverse")


def update_gaussian():
    update_image2(gaussian, "gaussian")
    print("Gata: Gaussian")


def update_vignette():
    update_image(vignette, 'vignette')
    print("Gata: Vignette")


def update_red_remove():
    update_image(red_remove, 'red_remove')
    print("Gata: Red Remove")


def update_green_remove():
    update_image(green_remove, 'green_remove')
    print("Gata: Green Remove")


def update_blue_remove():
    update_image(blue_remove, 'blue_remove')
    print("Gata: Blue Remove")


def update_smooth():
    update_image2(smooth, 'smooth')
    print("Gata: Smooth")


def update_smooth_10():
    update_image4(smooth, 'smooth')
    update_image4(smooth, 'smooth')
    update_image4(smooth, 'smooth')
    update_image4(smooth, 'smooth')
    update_image4(smooth, 'smooth')
    update_image4(smooth, 'smooth')
    update_image4(smooth, 'smooth')
    update_image4(smooth, 'smooth')
    update_image4(smooth, 'smooth')
    update_image2(smooth, 'smooth')
    print("Gata: Smooth x10")


def update_sharpen():
    update_image2(sharpen, 'sharpen')
    print("Gata: Sharpen")


def update_sharpen_10():
    update_image4(sharpen, 'sharpen')
    update_image4(sharpen, 'sharpen')
    update_image4(sharpen, 'sharpen')
    update_image4(sharpen, 'sharpen')
    update_image4(sharpen, 'sharpen')
    update_image4(sharpen, 'sharpen')
    update_image4(sharpen, 'sharpen')
    update_image4(sharpen, 'sharpen')
    update_image4(sharpen, 'sharpen')
    update_image2(sharpen, 'sharpen')
    print("Gata: Sharpen x10")


def update_mean():
    update_image2(mean, 'mean')
    print("Gata: Mean")


def update_mean_10():
    update_image4(mean, 'mean')
    update_image4(mean, 'mean')
    update_image4(mean, 'mean')
    update_image4(mean, 'mean')
    update_image4(mean, 'mean')
    update_image4(mean, 'mean')
    update_image4(mean, 'mean')
    update_image4(mean, 'mean')
    update_image4(mean, 'mean')
    update_image2(mean, 'mean')
    print("Gata: Mean x10")


def update_emboss():
    update_image2(emboss, 'emboss')
    print("Gata: Embross")


def update_emboss_10():
    update_image4(emboss, 'emboss')
    update_image4(emboss, 'emboss')
    update_image4(emboss, 'emboss')
    update_image4(emboss, 'emboss')
    update_image4(emboss, 'emboss')
    update_image4(emboss, 'emboss')
    update_image4(emboss, 'emboss')
    update_image4(emboss, 'emboss')
    update_image4(emboss, 'emboss')
    update_image2(emboss, 'emboss')
    print("Gata: Embross")


def contrast_p_128():
    update_image3(contrast, 'contrast', 128)
    print("Gata: Contrast 128")


def contrast_60():
    update_image3(contrast, 'contrast', 60)
    print("Gata: Contrast 60")


def update_sobel():
    update_image2(sobel, 'sobel')
    print("Gata: Sobel")


def update_prewitt():
    update_image2(prewitt, 'prewitt')
    print("Gata: Prewitt")


def update_GSSM():
    update_image4(gaussian, "gaussian")
    update_image4(smooth, 'smooth')
    update_image4(sharpen, 'sharpen')
    update_image2(mean, 'mean')
    print("Gata: GSSM")