import sys
import cv2
import requests
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from tkinter import messagebox
from tkinter import filedialog
from object_detection import ObjectDetection

# Due to tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
        if file_path == '':
            messagebox.showerror(
                "Error", "Please choose File")
            return
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
        if messagebox.askokcancel("Quit", "Do you really wish to quit?"):
            sys.exit()
        else:
            pass


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
        # img = np.asarray(img)
        print(handPredictions)
        for hand_pred_ret in handPredictions:
            prob = hand_pred_ret['probability']
            tagID = hand_pred_ret['tagId']
            if tagID == 0 and prob > 0.2:
                print(f"{hand_pred_ret['tagName']} {prob}")
                bbox = hand_pred_ret['boundingBox']
                left = bbox['left']
                top = bbox['top']
                width = bbox['width']
                height = bbox['height']
                x1 = int(left * img.shape[1])
                y1 = int(top * img.shape[0])
                x2 = x1 + int(width * img.shape[1])
                y2 = y1 + int(height * img.shape[0])
                cv2.imshow('test', img[y1:y2, x1:x2])
                sealPredictions = self.__isSealJudge(
                    Image.fromarray(img[y1:y2, x1:x2]))
                for seal_pred_ret in sealPredictions:
                    seal_prob = seal_pred_ret['probability']
                    seal_tagID = seal_pred_ret['tagId']
                    if seal_tagID == 0 and seal_prob > 0.2:
                        print(f"{seal_pred_ret['tagName']} {prob}")
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
