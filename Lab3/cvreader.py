import cv2
import numpy as np


class CVReader:
    def __init__(self, image_path):
        # Загружаем изображение и переводим его в цветовое пространство HSV
        self.image = cv2.imread(image_path)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.origin_x = 0
        self.origin_y = 0

    # Метод для поиска пересечения осей
    def find_axis_intersection(self):
        # Диапазоны для черного цвета (оси)
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 50])

        # Маска для черного цвета (оси)
        black_mask = cv2.inRange(self.hsv, black_lower, black_upper)

        # Поиск контуров для пересечения осей
        contours_black, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        for contour in contours_black:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    self.origin_x = int(M["m10"] / M["m00"])
                    self.origin_y = int(M["m01"] / M["m00"])

        return self.origin_x, self.origin_y

    def get_coordinates(self, color='blue', step=138):
        coordinates = []

        # Определяем диапазоны для цветов
        if color == 'blue':
            lower = np.array([100, 150, 50])
            upper = np.array([140, 255, 255])
        elif color == 'red':
            lower1 = np.array([0, 100, 100])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([160, 100, 100])
            upper2 = np.array([180, 255, 255])
            mask = cv2.bitwise_or(cv2.inRange(self.hsv, lower1, upper1), cv2.inRange(self.hsv, lower2, upper2))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    rel_x = cX - self.origin_x
                    rel_y = self.origin_y - cY
                    coordinates.append((rel_x / step, rel_y / step, 1))  # Добавляем кортеж вместо списка
            return coordinates

        # Маска для синего цвета (нулевой класс)
        blue_mask = cv2.inRange(self.hsv, lower, upper)

        # Поиск контуров для синих кругов (нулевой класс)
        contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_blue:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                rel_x = cX - self.origin_x
                rel_y = self.origin_y - cY
                coordinates.append((rel_x / step, rel_y / step, 0))  # Добавляем кортеж вместо списка

        return coordinates
