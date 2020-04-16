import sys
import tkinter as tk
from PIL import Image
from llsb_utils import *


def image_main(image_filename):

    od_model = TFObjectDetection(model.hand_graph_def, model.is_hand_labels)

    image = Image.open(image_filename)
    predictions = od_model.predict_image(image)
    print(predictions)


if __name__ == '__main__':
    model = Models()
    startpage = OptionFrame(model=model)
    # model.imageJydge('/Users/lulusabee/Documents/Codehere/Check-NKUST-Seal/images/test-Image/test-4.jpg')
