import sys
import tkinter as tk
from tkinter import filedialog


def main():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    folder = filedialog.askdirectory()

    root.destroy()

    if folder:
        print(folder)
        sys.exit(0)

    sys.exit(1)


if __name__ == "__main__":
    main()
