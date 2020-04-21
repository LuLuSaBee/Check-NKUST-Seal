import sys
import cv2
import requests
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk
from PIL import Image
from io import BytesIO
from tkinter import messagebox
from tkinter import filedialog
from object_detection import ObjectDetection

# Due to tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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


class ActionHandler():
    def __init__(self):
        super().__init__()
        self.magistrate = Magistrate()
        self.img_formats = ("image/png", "image/jpeg", "image/gif", "image/jpg",
                            "image/tiff", "image/bmp", "image/x-xpm", "image/webp")

    def btnClick(self, action, url=None, deviceID=None):
        if action == "OpenImg":
            self.__getFile()
        elif action == "UrlImg":
            self.__getUrlImg(url)
        elif action == "OpenDev":
            self.__openWebcam(int(deviceID))
        elif action == "Exit":
            self.__askQuit()

    def __getFile(self):
        file_path = filedialog.askopenfilename()
        try:
            img = Image.open(file_path)
            self.magistrate.showImage(img=img)
        except Image.UnidentifiedImageError:
            messagebox.showerror(
                "Error", "File must be \"Image\"")

    def __getUrlImg(self, url):
        string = ""
        try:
            response = requests.get(url, timeout=3)
            if response.headers['content-type'] in self.img_formats:
                img = Image.open(BytesIO(response.content))
                self.magistrate.showImage(img=img)
                return
            else:
                string = "is not image"
        except requests.exceptions.HTTPError as errh:
            string = f"Http Error:{errh}"
        except requests.exceptions.ConnectionError as errc:
            string = f"Error Connecting:{errc}"
        except requests.exceptions.Timeout as errt:
            string = f"Timeout Error:{errt}"
        except requests.exceptions.RequestException as err:
            string = f"OOps: Something Else:{err}"
        messagebox.showerror("Error", string)

    def __openWebcam(self, deviceID):
        cap = cv2.VideoCapture(deviceID)
        self.magistrate.showImage(cap=cap)
        cap.release()
        cv2.destroyAllWindows()

    def __askQuit(self):
        # if messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        sys.exit()


class Magistrate():
    def __init__(self):
        super().__init__()
        self.model = Models()

    def showImage(self, img=None, cap=None):
        if cap != None:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            scale_percent = 600 / height
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width * scale_percent))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height * scale_percent))
        else:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            height, width, c = img.shape
            scale_percent = 600 / height
            newsize = (int(width * scale_percent), int(height * scale_percent))
            img = cv2.resize(img, newsize)

        while True:
            if cap != None:
                ret, frame = cap.read()
                if not ret:
                    messagebox.showerror(
                        "Error", "VideoCapture.read() failed, Exiting...")
                    break
                img = frame

            verdict, fontcolor = self.__judge(img)
            cv2.putText(img, verdict,
                        (50, 50), cv2.FONT_ITALIC, 2, fontcolor, 4)
            cv2.imshow('Checker', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            else:
                continue

    def __judge(self, img):
        handPredictions = self.__isHandJudge(Image.fromarray(img))
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        for hand_pred_ret in handPredictions:
            prob = hand_pred_ret['probability']
            tagID = hand_pred_ret['tagId']
            if tagID == 0 and prob > 0.2:
                print("{tagID} {prob}")
                bbox = hand_pred_ret['boundingBox']
                left = bbox['left']
                top = bbox['top']
                width = bbox['width']
                height = bbox['height']
                x1 = int(left * img.shape[1])
                y1 = int(top * img.shape[0])
                x2 = x1 + int(width * img.shape[1])
                y2 = y1 + int(height * img.shape[0])
                isHand = True
                sealPredictions = self.__isSealJudge(Image.fromarray(img))
                for seal_pred_ret in sealPredictions:
                    seal_prob = seal_pred_ret['probability']
                    seal_tagID = seal_pred_ret['tagId']
                    if seal_tagID == 0 and seal_prob > 0.2:
                        print("{tagID} {prob}")
                        return "Pass", (0, 255, 0)

        return "None", (0, 0, 0)

    def __isHandJudge(self, image):
        hand_model = TFObjectDetection(self.model.get_Hand_ModelandLabel())

        predictions = hand_model.predict_image(image)
        return predictions

    def __isSealJudge(self, image):
        seal_model = TFObjectDetection(self.model.get_Seal_ModelandLabel())

        predictions = seal_model.predict_image(image)
        return predictions


class Models():
    __is_hand_model_path = './resource/is_hand_model.pb'
    __is_hand_label_path = './resource/is_hand_labels.txt'
    __have_seal_model_path = './resource/have_seal_model.pb'
    __have_seal_label_path = './resource/have_seal_labels.txt'
    __hand_graph_def = tf.compat.v1.GraphDef()
    __seal_graph_def = tf.compat.v1.GraphDef()

    def __init__(self):
        super().__init__()
        # Load a TensorFlow model
        with tf.io.gfile.GFile(self.__is_hand_model_path, 'rb') as f:
            self.__hand_graph_def.ParseFromString(f.read())
        with tf.io.gfile.GFile(self.__have_seal_model_path, 'rb') as f:
            self.__seal_graph_def.ParseFromString(f.read())

        # Load labels
        with open(self.__is_hand_label_path, 'r') as f:
            self.__hand_labels = [l.strip() for l in f.readlines()]
        with open(self.__have_seal_label_path, 'r') as f:
            self.__seal_labels = [l.strip() for l in f.readlines()]

    def get_Hand_ModelandLabel(self):
        return dict({'graph_def': self.__hand_graph_def, 'labels': self.__hand_labels})

    def get_Seal_ModelandLabel(self):
        return dict({'graph_def': self.__seal_graph_def, 'labels': self.__seal_labels})


class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow"""

    def __init__(self, modelinfo):
        super(TFObjectDetection, self).__init__(modelinfo['labels'])
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            input_data = tf.compat.v1.placeholder(
                tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(modelinfo['graph_def'], input_map={
                                "Placeholder:0": input_data}, name="")

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float)[
            :, :, (2, 1, 0)]  # RGB -> BGR

        with tf.compat.v1.Session(graph=self.graph) as sess:
            output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
            outputs = sess.run(
                output_tensor, {'Placeholder:0': inputs[np.newaxis, ...]})
            return outputs[0]
