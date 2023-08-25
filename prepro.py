import cv2
import numpy as np

class Prepro:
    def __init__(self):
        self.image_size = (420,280,3)

    def make_coordinates(self,image, line_parameters):
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1*(3/5))
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return np.array([x1, y1, x2, y2])

    def average_slope_intercept(self,image, lines):
        left_fit = []
        right_fit = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = self.make_coordinates(image, left_fit_average)
        right_line = self.make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])

    def canny(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray,(5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)
        return canny

    def display_lines(self,image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_image

    def region_of_interest(self,image):
        height = image.shape[0]
        polygons = np.array([
        [(0, height), (420, height), (225, 150)]
        ])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
    
    def processing(self,image):
        lane_image = np.copy(image)
        canny_image = self.canny(lane_image)
        lines = cv2.HoughLinesP(canny_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = self.average_slope_intercept(lane_image,lines)
        line_image  = self.display_lines(lane_image,averaged_lines)
        combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)
        print(combo_image.shape)
        combo_image = np.reshape(combo_image,(280,420,4))
        combo_image = combo_image[:,:,:3][:] #np.array
        images = []
        images.append(combo_image)
        input = np.array(images)
    


        return input


