import cv2
import tkinter as tk
from tkinter import ttk
from llsb_utils import ActionHandler


class GUI():
    def __init__(self, width=200, height=130, textFont=('Hack', 24)):
        super().__init__()
        handler = ActionHandler()
        self = tk.Tk()
        self.position_x = (self.winfo_screenwidth()/2) - (width*2)
        self.position_y = (self.winfo_screenheight()/2) - (height*2)
        self.title('NKUST Seal Checker')
        self.geometry('%dx%d+%d+%d' %
                      (width, height, self.position_x, self.position_y))
        self.configure(background='white')

        def openInputUrlForm():
            form = tk.Tk()
            form.title("Input URL")
            form.geometry("+300+300")
            form_entry = tk.Entry(form)
            form_entry.focus()
            form_entry.pack()
            btn_frame = tk.Frame(form)
            btn_frame.pack()
            form_subbtn = tk.Button(btn_frame, text='   Enter   ', font=('Hack', 12),
                                    command=lambda: [f() for f in [lambda: handler.btnClick(
                                        "UrlImg", form_entry.get()), form.destroy]])

            form_exitbtn = tk.Button(btn_frame, text='  Cancel  ',
                                     font=('Hack', 12), command=form.destroy)
            form_subbtn.pack(side=tk.LEFT)
            form_exitbtn.pack(side=tk.LEFT)
            form.mainloop()

        def deviceList():
            index = 0
            webcam_list = []
            while True:
                if not cv2.VideoCapture(index).read()[0]:
                    break
                else:
                    webcam_list.append(index)
                index += 1
            if index == 0:
                messagebox.showerror(
                    "Device Error", "You don't have Webcam device")
                return
            form = tk.Tk()
            form.title("Choose DeviceID")
            form.geometry("+400+400")
            form_combo = ttk.Combobox(form, values=webcam_list)
            form_combo.current(0)
            form_combo.pack()
            btn_frame = tk.Frame(form)
            btn_frame.pack()
            form_subbtn = tk.Button(btn_frame, text='   Enter   ', font=('Hack', 12),
                                    command=lambda: [f() for f in [lambda:
                                                                   handler.btnClick(
                                                                       "OpenDev", deviceID=form_combo.get()),
                                                                   form.destroy]])
            form_exitbtn = tk.Button(btn_frame, text='  Cancel  ',
                                     font=('Hack', 12), command=form.destroy)
            form_subbtn.pack(side=tk.LEFT)
            form_exitbtn.pack(side=tk.LEFT)
            form.mainloop()

        self.choose_img_btn = tk.Button(
            self, text='Choose Image', font=textFont, command=lambda: handler.btnClick("OpenImg"))
        self.url_img_btn = tk.Button(
            self, text='Open  ImgUrl', font=textFont, command=openInputUrlForm)
        self.open_dev_btn = tk.Button(
            self, text='Open  Webcam', font=textFont, command=deviceList)
        self.exit_btn = tk.Button(
            self, text='    Exit    ', font=textFont, command=lambda: handler.btnClick("Exit"))

        self.choose_img_btn.pack()
        self.url_img_btn.pack()
        self.open_dev_btn.pack()
        self.exit_btn.pack()
        self.protocol("WM_DELETE_WINDOW", lambda: handler.btnClick("Exit"))
        self.mainloop()


if __name__ == '__main__':
    mygui = GUI()
