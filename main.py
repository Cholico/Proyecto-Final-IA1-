import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython import display

resultados_x = []
resultados_y = []
def NCC(img, temp, x, y):
    H = temp.shape[0]
    W = temp.shape[1]

    sum_img = 0.0
    sum_temp = 0.0
    sum_2 = 0.0

    for i in range(W):
        for j in range(H):
            sum_img = sum_img + float(img[int(y)+j, int(x)+i]) ** 2
            sum_temp = sum_temp + float(temp[j, i]) ** 2
            sum_2 = sum_2 + float(img[int(y)+j, int(x)+i]) * float(temp[j, i])

    val = sum_2 / (np.sqrt(float(sum_img)) * np.sqrt(float(sum_temp)))

    return val

def DE(img, temp, animacion):
    n_Gen = 100
    n_Pop = 50
    dim = 2

    F = 0.8
    Cr = 0.5

    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp_g = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    img_H = img_g.shape[0]
    img_W = img_g.shape[1]
    temp_H = temp_g.shape[0]
    temp_W = temp_g.shape[1]

    lb = np.array([1, 1])
    ub = np.array([img_W-temp_W, img_H-temp_H])

    x = np.zeros((dim, n_Pop))
    v = np.zeros((dim, n_Pop))
    u = np.zeros((dim, n_Pop))
    val_max = float('-inf')
    fitness = -1 * np.ones(n_Pop)

    for i in range(n_Pop):
        x[0, i] = np.random.randint(lb[0], ub[0])
        x[1, i] = np.random.randint(lb[1], ub[1])

        fitness[i] = NCC(img_g, temp_g, int(x[0, i]), int(x[1, i]))

    best_plot = np.zeros(n_Gen)

    for n in range(n_Gen):
        if animacion:
            img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.clf()
            display.display(plt.gcf())
            display.clear_output(wait=True)
            plt.imshow(img_2)

            for i in range(n_Pop):
                plt.plot([x[0, i], x[0, i] + temp_W], [x[1, i], x[1, i]], color=(0,1,0), linewidth=3)
                plt.plot([x[0, i], x[0, i]], [x[1, i], x[1, i] + temp_H], color=(0,1,0), linewidth=3)
                plt.plot([x[0, i] + temp_W, x[0, i] + temp_W], [x[1, i], x[1, i] + temp_H], color=(0,1,0), linewidth=3)
                plt.plot([x[0, i], x[0, i] + temp_W], [x[1, i] + temp_H, x[1, i] + temp_H], color=(0,1,0), linewidth=3)
            plt.show(block=False)
            plt.pause(.05)
            #plt.show()

        for i in range(n_Pop):
            # Selección de vectores de muestra
            r1 = i
            while r1 == i:
                r1 = np.random.randint(n_Pop)

            r2 = r1
            while r2 == r1 or r2 == i:
                r2 = np.random.randint(n_Pop)

            r3 = r2
            while r3 == r2 or r3 == r1 or r3 == i:
                r3 = np.random.randint(n_Pop)

            v[:, i] = np.floor(x[:, r1] + F * (x[:, r2] - x[:, r3]))

            ra = np.random.rand()

            for j in range(dim):
                if ra <= Cr:
                    u[j, i] = v[j, i]
                else:
                    u[j, i] = x[j, i]


        # Cálculo de Penalización
            for j in range(dim):
                if lb[j] <= u[j, i] <= ub[j]:
                    u[j, i] = u[j, i]
                else:
                    u[j, i] = (lb[j] + np.floor(ub[j] - lb[j]) * np.random.rand())


            if NCC(img_g, temp_g, u[0, i], u[1, i]) > NCC(img_g, temp_g, x[0, i], x[1, i]):
                x[:, i] = u[:, i]

            if NCC(img_g, temp_g, x[0, i], x[1, i]) > val_max:
                val_max = NCC(img_g, temp_g, x[0, i], x[1, i])
                xp, yp = x[0, i], x[1, i]


        best_plot[n] = np.max(fitness)
        resultados_x = best_plot.copy()

    ind = np.argmax(fitness)
    p = x[:, ind]

    plt.plot(best_plot)
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.show()

    return p

animacion = 1

img = cv2.imread('Image_1.bmp')
temp = cv2.imread('Template.bmp')

p = DE(img, temp, animacion)

img_H = img.shape[0]
img_W = img.shape[1]
temp_H = temp.shape[0]
temp_W = temp.shape[1]

xp = p[0]
yp = p[1]

for i in resultados_x:
    print(i)

img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_2)
plt.plot([xp, xp + temp_W], [yp, yp], 'r', linewidth=4)
plt.plot([xp, xp], [yp, yp + temp_H], 'r', linewidth=4)
plt.plot([xp + temp_W, xp + temp_W], [yp, yp + temp_H], 'r', linewidth=4)
plt.plot([xp, xp + temp_W], [yp + temp_H, yp + temp_H], 'r', linewidth=4)
plt.show()


