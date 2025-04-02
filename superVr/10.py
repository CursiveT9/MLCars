import cv2
import numpy as np

def detect_and_draw_markers(frame):
    """Обнаруживает и выделяет маркеры ArUCO на кадре видеопотока"""
    # Инициализация детектора ArUCO
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Детектирование маркеров
    corners, ids, rejected = detector.detectMarkers(frame)

    # Рисование обнаруженных маркеров
    if ids is not None:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            # Извлекаем углы маркера
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # Преобразуем координаты в целые числа
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # Рисуем контур маркера
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

            # Вычисляем и рисуем центр маркера
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

            # Подписываем ID маркера
            cv2.putText(frame, str(markerID),
                        (topLeft[0], topLeft[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    return frame

def main():
    # Захват видео с камеры (по умолчанию используется первая камера, если их несколько, можно указать номер)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Не удалось открыть видеокамеру.")
        return

    while True:
        # Чтение кадра с видеокамеры
        ret, frame = cap.read()
        if not ret:
            print("Не удалось захватить кадр.")
            break

        # Обработка кадра
        frame_with_markers = detect_and_draw_markers(frame)

        # Отображение кадра с маркерами
        cv2.imshow("ArUco Marker Detection", frame_with_markers)

        # Выход из цикла по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
