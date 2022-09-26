from network import Network
from PIL import Image
import cv2
import numpy as np
def load_images():
    images = []
    img_name = '/home/min/sc/mydata2/mydata/1663523534.03.jpg'
    image = cv2.imread(img_name)
    image = cv2.resize(image,(320,180))
    images.append(image)
    return np.array(images)

def main():
    global image
    model = Network()
    #image = np.reshape(image,(280,420,4))
    test_img = load_images()
    my_angle = model.angle_predict(test_img)
    predict = my_angle[0][0]
    print(predict)
##성공##
main()