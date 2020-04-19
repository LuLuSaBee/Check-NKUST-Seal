from tkinter import filedialog
from llsb_utils import *
import sys
from tkinter import *


class GUI(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.grid()

        self.fnameLabel = Label(master, text="First Name")
        self.fnameLabel.grid()

        self.fnameEntry = Entry(master)
        self.fnameEntry.grid()

        self.lnameLabel = Label(master, text="Last Name")
        self.lnameLabel.grid()

        self.lnameEntry = Entry(master)
        self.lnameEntry.grid()

        self.submitButton = Button(
            master, command=self.buttonClick, text="Submit")
        self.submitButton.grid()

    def buttonClick(self):
        return "Hello"


if __name__ == "__main__":
    guiFrame = GUI()
    print(guiFrame.submitButton.cget("command"))
    guiFrame.mainloop()


image_file_path = ""


def initGUI(width=200, height=100, textFont=('Hack', 24)):
    position_x = (window.winfo_screenwidth()/2) - (width*2)
    position_y = (window.winfo_screenheight()/2) - (height*2)
    window.title('NKUST Seal Checker')
    window.geometry('%dx%d+%d+%d' % (width, height,  position_x,  position_y))
    window.configure(background='white')
    choose_img_btn = Button(window, text='Choose Image',
                            font=textFont, command=openFile)
    choose_img_btn.pack()
    open_dev_btn = Button(window, text='Open  Webcam',
                          font=textFont)
    open_dev_btn.pack()
    exit_btn = Button(window, text='    Exit    ',
                      font=textFont, command=askQuit)
    exit_btn.pack()


def openFile():
    image_file_path = filedialog.askopenfilename()


if __name__ == '__main__':
    # main()
    model = Models()
    window = Tk()
    initGUI()
    window.mainloop()
    print(image_file_path)
