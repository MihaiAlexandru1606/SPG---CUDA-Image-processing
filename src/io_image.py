from PIL import ImageTk, Image
from tkinter import Label
from src.global_var import GlobalVar
import cupy as cp
from copy import deepcopy


def read_image(file_path):
    img = Image.open(file_path, "r")
    GlobalVar.width, GlobalVar.height = img.size
    GlobalVar.pixels = cp.array(img.getdata(), dtype=cp.int32)
    GlobalVar.original_img = deepcopy(GlobalVar.pixels)

    new_size = str(str(GlobalVar.width) + "x" + str(GlobalVar.height))
    GlobalVar.window.geometry(new_size)
    GlobalVar.window.resizable(width=True, height=True)

    if not GlobalVar.is_image:
        tk_img = ImageTk.PhotoImage(img)
        GlobalVar.panel_img = Label(GlobalVar.window, image=tk_img)
        GlobalVar.panel_img.pack(side="bottom", fill="both", expand="yes")
        GlobalVar.is_image = True
        GlobalVar.panel_img.mainloop()
    else:
        tk_img = ImageTk.PhotoImage(img)
        GlobalVar.panel_img.configure(image=tk_img)
        GlobalVar.panel_img.image = tk_img


def save_image(name: str):
    new_data = cp.asnumpy(GlobalVar.pixels).reshape(GlobalVar.height * GlobalVar.width, 3)
    new_data = [tuple(x) for x in new_data]

    img = Image.new(mode="RGB", size=(GlobalVar.width, GlobalVar.height))
    img.putdata(new_data)
    img.save(name)


def display_new_image():
    # acum se face update la imagine
    new_data = cp.asnumpy(GlobalVar.pixels).reshape(GlobalVar.height * GlobalVar.width, 3)
    new_data = [tuple(x) for x in new_data]

    new_img = Image.new(mode="RGB", size=(GlobalVar.width, GlobalVar.height))
    new_img.putdata(new_data)
    tk_img = ImageTk.PhotoImage(new_img)

    GlobalVar.panel_img.configure(image=tk_img)
    GlobalVar.panel_img.image = tk_img
