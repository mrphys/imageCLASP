# import sys
# import tkinter as tk
# from tkinter import filedialog


# def main():
#     root = tk.Tk()
#     root.withdraw()
#     root.attributes("-topmost", True)

#     folder = filedialog.askdirectory()

#     root.destroy()

#     if folder:
#         print(folder)
#         sys.exit(0)

#     sys.exit(1)


# if __name__ == "__main__":
#     main()

import wx
 
def pick_folder():
    app = wx.App(False)
    frame = wx.Frame(None)
 
    with wx.DirDialog(
        frame,
        message="Choose a folder",
        style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST
    ) as dialog:
        if dialog.ShowModal() == wx.ID_OK:
            return dialog.GetPath()
    return None
 
if __name__ == "__main__":
    folder = pick_folder()
    print(folder)