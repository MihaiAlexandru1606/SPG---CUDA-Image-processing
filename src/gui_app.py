from tkinter import *
from tkinter import filedialog
from src.global_var import GlobalVar
from src.io_image import read_image, save_image
from src.image_processing import *
import cupy as cp
import os
from copy import deepcopy


def get_path_to_image():
    """
    functia returneaza calea catre imagine, unde este citita
    """
    file_types = [
        ("JPEG Image", '*.jpeg; *jpg'),
        ("PNG Image", '*.png'),
        ("BPM Image", '*.bmp'),
        ("Netpbm Image", '*.ppm; *.pgm; *.pbm; *pnm')
    ]

    GlobalVar.file_path = filedialog.askopenfilename(filetypes=file_types)
    GlobalVar.name_original = GlobalVar.file_path.split('/')[-1]
    GlobalVar.is_open_image = True

    read_image(GlobalVar.file_path)


def save_normal():
    if not os.path.exists('../procsesing_folder'):
        os.makedirs('../procsesing_folder')

    name = '../procsesing_folder/' + GlobalVar.name_operation + "_" + GlobalVar.name_original
    save_image(name)


def save_as():
    folder_selected = filedialog.askdirectory()
    name = str(folder_selected) + "/" + GlobalVar.name_operation + "_" + GlobalVar.name_original
    save_image(name)


def back_to_original():
    if not GlobalVar.is_image:
        return

    GlobalVar.pixels = deepcopy(GlobalVar.original_img)

    # acum se face update la imagine
    new_data = cp.asnumpy(GlobalVar.pixels).reshape(GlobalVar.height * GlobalVar.width, 3)
    new_data = [tuple(x) for x in new_data]

    new_img = Image.new(mode="RGB", size=(GlobalVar.width, GlobalVar.height))
    new_img.putdata(new_data)
    tk_img = ImageTk.PhotoImage(new_img)

    GlobalVar.panel_img.configure(image=tk_img)
    GlobalVar.panel_img.image = tk_img


def run_app():
    GlobalVar.window = Tk()
    GlobalVar.window.title("CUDA : Image processing")
    GlobalVar.window.geometry("400x500")
    menu = Menu(GlobalVar.window)

    file_menu = Menu(menu)

    file_menu.add_command(label='Open', command=get_path_to_image)
    file_menu.add_command(label='Save', command=save_normal)  # momenta doar are rolul de varificare
    file_menu.add_command(label='Save as', command=save_as)
    file_menu.add_separator()
    file_menu.add_command(label='Back to Original', command=back_to_original)
    file_menu.add_separator()
    file_menu.add_command(label='Exit', command=GlobalVar.window.quit)

    simple_operation = Menu(menu)
    simple_operation.add_command(label='GrayScale', command=update_grayscale)
    simple_operation.add_separator()
    simple_operation.add_command(label='Red Only', command=update_red_only)
    simple_operation.add_command(label='Green Only', command=update_green_only)
    simple_operation.add_command(label='Blue Only', command=update_blue_only)
    simple_operation.add_separator()
    simple_operation.add_command(label='Inverse', command=update_inverse)
    simple_operation.add_separator()
    simple_operation.add_command(label='Red Remove', command=update_red_remove)
    simple_operation.add_command(label='Green Remove', command=update_green_remove)
    simple_operation.add_command(label='Blue Remove', command=update_blue_remove)
    simple_operation.add_separator()
    simple_operation.add_command(label='Contrast 128', command=contrast_p_128)
    simple_operation.add_command(label='Contrast 60', command=contrast_60)

    filter_menu = Menu(menu)
    filter_menu.add_command(label='Gaussian', command=update_gaussian)
    filter_menu.add_command(label='Vignette', command=update_vignette)

    filter_menu.add_command(label='Smooth', command=update_smooth)
    filter_menu.add_command(label='Smooth 10x', command=update_smooth_10)

    filter_menu.add_command(label='Sharpen', command=update_sharpen)
    filter_menu.add_command(label='Sharpen 10x', command=update_sharpen_10)

    filter_menu.add_command(label='Mean', command=update_mean)
    filter_menu.add_command(label='Mean 10x', command=update_mean_10)

    filter_menu.add_command(label='Emboss', command=update_emboss)
    filter_menu.add_command(label='Emboss 10x', command=update_emboss_10)

    filter_menu.add_separator()
    filter_menu.add_command(label='GSSM', command=update_GSSM)

    edge_detection = Menu(menu)
    edge_detection.add_command(label='Sobel', command=update_sobel)
    edge_detection.add_command(label='Prewitt', command=update_prewitt)

    menu.add_cascade(label='File', menu=file_menu)
    menu.add_cascade(label='Simple Operation', menu=simple_operation)
    menu.add_cascade(label='Filter', menu=filter_menu)
    menu.add_cascade(label='Edge Detection', menu=edge_detection)

    GlobalVar.window.config(menu=menu)
    GlobalVar.window.mainloop()


if __name__ == '__main__':
    run_app()
