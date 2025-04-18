import numpy as np
from PIL import Image


def load_and_preprocess_image(image_path):
    """ Загружает изображение и преобразует его в бинарный массив (ч/б для каждого цвета) """
    img = Image.open(image_path).convert("RGB")  # Загружаем как цветное
    img_np = np.array(img)

    # Создаём маску (True - цветной объект, False - фон)
    mask = np.any(img_np != [255, 255, 255], axis=-1)

    return mask.astype(np.uint8) * 255  # 255 - объекты, 0 - фон


def find_objects(image):
    """Находит все объекты (круги и квадраты)"""
    visited = np.zeros_like(image, dtype=bool)
    objects = []

    def bfs(x, y):
        """Обходит объект и собирает все его пиксели"""
        queue = [(x, y)]
        pixels = []
        while queue:
            cx, cy = queue.pop()
            if visited[cy, cx]:
                continue
            visited[cy, cx] = True
            pixels.append((cx, cy))

            # Проверяем 4 соседних пикселя
            for nx, ny in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]:
                if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                    if image[ny, nx] == 255 and not visited[ny, nx]:
                        queue.append((nx, ny))

        return pixels

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] == 255 and not visited[y, x]:
                objects.append(bfs(x, y))

    return objects


def classify_objects(objects):
    """Определяет, какие из объектов — круги, а какие — квадраты."""
    circles, squares = 0, 0

    for obj in objects:
        xs, ys = zip(*obj)
        width, height = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        area = len(obj)
        bounding_box_area = width * height
        fill_ratio = area / bounding_box_area  # Заполняемость

        if fill_ratio > 0.85:  # Если объект плотно заполняет bounding box → квадрат
            squares += 1
        elif fill_ratio > 0.65:  # Если объект менее плотный → круг
            circles += 1

    return circles, squares


def count_shapes(image_path):
    """Загружает изображение и считает круги и квадраты"""
    image = load_and_preprocess_image(image_path)
    objects = find_objects(image)
    circles, squares = classify_objects(objects)
    return circles, squares


# Проверка
image_path = "222.png"
circles, squares = count_shapes(image_path)
print(f"Круги: {circles}, Квадраты: {squares}")
