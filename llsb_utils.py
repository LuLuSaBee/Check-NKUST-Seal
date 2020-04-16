import sys
import tkinter as tk
import numpy as np
import tensorflow as tf
from tkinter import filedialog
from PIL import Image
from object_detection import ObjectDetection
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class OptionFrame():
    window = tk.Tk()

    def __init__(self, model, width=200, height=100, textFont=('Hack', 24)):
        super().__init__()
        self.position_x = (self.window.winfo_screenwidth()/2) - (width*2)
        self.position_y = (self.window.winfo_screenheight()/2) - (height*2)
        self.window.title('NKUST Seal Checker')
        self.window.geometry('%dx%d+%d+%d' %
                             (width, height, self.position_x, self.position_y))
        self.window.configure(background='white')
        self.choose_img_btn = tk.Button(
            self.window, text='Choose Image', font=textFont, command=lambda: self.openFile(model))
        self.choose_img_btn.pack()
        self.open_dev_btn = tk.Button(
            self.window, text='Open  Webcam', font=textFont)
        self.open_dev_btn.pack()
        self.exit_btn = tk.Button(
            self.window, text='    Exit    ', font=textFont, command=self.quit)
        self.exit_btn.pack()
        self.window.mainloop()

    def disAllBtn(self):
        self.choose_img_btn.config(state=tk.DISABLED)
        self.open_dev_btn.config(state=tk.DISABLED)
        self.exit_btn.config(state=tk.DISABLED)

    def actAllBtn(self):
        self.choose_img_btn.config(state=tk.ACTIVE)
        self.open_dev_btn.config(state=tk.ACTIVE)
        self.exit_btn.config(state=tk.ACTIVE)

    def openFile(self, model):
        _ = tk.Tk()
        _.withdraw()

        file_path = filedialog.askopenfilename()
        model.imageJydge(file_path)

    def quit(self):
        sys.exit()


class Models():
    __is_hand_model_path = './resource/is_hand_model.pb'
    __is_hand_label_path = './resource/is_hand_labels.txt'
    __have_seal_model_path = './resource/have_seal_model.pb'
    __have_seal_label_path = './resource/have_seal_labels.txt'
    hand_graph_def = tf.compat.v1.GraphDef()
    seal_graph_def = tf.compat.v1.GraphDef()

    def __init__(self):
        super().__init__()
        # Load a TensorFlow model
        with tf.io.gfile.GFile(self.__is_hand_model_path, 'rb') as f:
            self.hand_graph_def.ParseFromString(f.read())
        with tf.io.gfile.GFile(self.__have_seal_model_path, 'rb') as f:
            self.seal_graph_def.ParseFromString(f.read())

        # Load labels
        with open(self.__is_hand_label_path, 'r') as f:
            self.is_hand_labels = [l.strip() for l in f.readlines()]
        with open(self.__have_seal_label_path, 'r') as f:
            self.have_seal_labels = [l.strip() for l in f.readlines()]

    def imageJydge(self, file):
        od_model = TFObjectDetection(self.hand_graph_def, self.is_hand_labels)

        image = Image.open(file)
        predictions = od_model.predict_image(image)
        print(predictions)


class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow"""

    def __init__(self, graph_def, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            input_data = tf.compat.v1.placeholder(
                tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(graph_def, input_map={
                                "Placeholder:0": input_data}, name="")

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float)[
            :, :, (2, 1, 0)]  # RGB -> BGR

        with tf.compat.v1.Session(graph=self.graph) as sess:
            output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
            outputs = sess.run(
                output_tensor, {'Placeholder:0': inputs[np.newaxis, ...]})
            return outputs[0]
