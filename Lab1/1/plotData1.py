import matplotlib.pyplot as plt

def plotData(x, y):
    plt.scatter(x, y, c='red', label='Обучающие данные')
    plt.xlabel('Количество автомобилей')
    plt.ylabel('Прибыль СТО')
    plt.title('Зависимость прибыли от количества автомобилей')
    plt.legend()
    plt.grid(True)
    plt.show()
