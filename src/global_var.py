class GlobalVar(object):
    is_open_image = False
    file_path = None
    pixels = None
    width = None
    height = None
    window = None
    panel_img = None
    is_image = False
    name_original = None
    name_operation = ""

    original_img = None

    @staticmethod
    def reset_global_var(self):
        is_open_image = False
        file_path = None
        pixels = None
        width = None
        height = None

        name_original = None
        type_operation = None
