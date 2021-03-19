from Packages.GeoPlot import *
from Packages.RegistrationFunc import *
import numpy as np
import scipy.io as sio


def vis_ellipses(tensor_field, title=None):
    eps11 = tensor_field[0, :, :]
    eps12 = tensor_field[1, :, :]
    eps22 = tensor_field[2, :, :]
    # visualizing ellipses
    ells = []
    tens = np.zeros((2, 2))
    scale = 0.3
    for x in range(1, eps11.shape[0] - 1):
        for y in range(1, eps11.shape[1] - 1):
            # evals and evecs by numpy
            tens[0, 0] = eps11[x, y]
            tens[0, 1] = eps12[x, y]
            tens[1, 0] = eps12[x, y]
            tens[1, 1] = eps22[x, y]
            evals, evecs = np.linalg.eigh(tens)
            angles = np.degrees(np.math.atan2(evecs[1][1], evecs[1][0]))
            ells.append(Ellipse(xy=(x, y), width=scale * evals[1], height=scale * evals[0], angle=angles))
    # plt.figure(1)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'}, figsize=(8, 8))
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(1)
        e.set_facecolor([0, 0, 0])
    ax.set_xlim(0, eps11.shape[0])
    ax.set_ylim(0, eps11.shape[1])
    ax.set_title(title)
    ax.legend()
    # plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # N = 100
    # h, w, bezel = 145, 174, 10
    h, w, bezel = 100, 100, 1
    # ori_coord_h, ori_coord_w = 50, 50
    evals = [6, 5.69, 5.06, 4.75, 4.44, 4.12, 3.5]
    lambd1 = np.ones((h, w))
    lambd2 = np.ones((h, w))
    tensor = np.zeros((3, h, w))

    # tensor 1: y = 0.001(x-50)^3+50
    ori_coord_h, ori_coord_w = 72, 87
    for i in range(bezel, h - bezel):
        j = int(0.001 * (i - ori_coord_h) ** 3 + ori_coord_w)
        if j in range(w):
            lambd1[i, j] = evals[0]

    for j in range(bezel, ori_coord_w):
        i = int(-((ori_coord_w - j) * 1000) ** (1.0 / 3) + ori_coord_h)
        if i in range(h):
            lambd1[i, j] = evals[0]

    for j in range(ori_coord_w, w - bezel):
        i = int(((j - ori_coord_w) * 1000) ** (1.0 / 3) + ori_coord_h)
        if i in range(h):
            lambd1[i, j] = evals[0]

    # # tensor 2: y = 0.0001(x-ori_coord_h)^4+ori_coord_w
    # ori_coord_h, ori_coord_w = 72, 30
    # for i in range(h-bezel):
    #     j = int(0.0001 * (i - ori_coord_h) ** 4 + ori_coord_w)
    #     if j in range(w):
    #         lambd1[i,j] = 6
    #
    # for j in range(ori_coord_w,w-bezel):
    #     i = int(10*(j-ori_coord_w)**(1.0/4)+ori_coord_h)
    #     if i in range(w):
    #         lambd1[i,j] = 6
    #     i = int(-10*(j-ori_coord_w)**(1.0/4)+ori_coord_h)
    #     if i in range(w):
    #         lambd1[i,j] = 6

    # # tensor 3: y = 30sin(0.0005(x-ori_coord_h)^2)+ori_coord_w
    # ori_coord_h, ori_coord_w = 0, 50
    # for i in range(bezel,h-bezel):
    #     j = int(30 * np.sin(0.0005*(i-ori_coord_h)**2) +ori_coord_w)
    #     if j in range(w):
    #         lambd1[i,j] = 6

    # plt.figure()
    # plt.imshow(lambd1)
    # plt.show()

    for k in range(1, len(evals)):
        for i in range(bezel, h - bezel):
            for j in range(bezel, w - bezel):
                if lambd1[i, j] == evals[k - 1]:
                    if lambd1[i - 1, j] == 1:
                        lambd1[i - 1, j] = evals[k]

                    if lambd1[i + 1, j] == 1:
                        lambd1[i + 1, j] = evals[k]

                    if lambd1[i, j - 1] == 1:
                        lambd1[i, j - 1] = evals[k]

                    if lambd1[i, j + 1] == 1:
                        lambd1[i, j + 1] = evals[k]

                    if lambd1[i - 1, j - 1] == 1:
                        lambd1[i - 1, j - 1] = evals[k]

                    if lambd1[i - 1, j + 1] == 1:
                        lambd1[i - 1, j + 1] = evals[k]

                    if lambd1[i + 1, j - 1] == 1:
                        lambd1[i + 1, j - 1] = evals[k]

                    if lambd1[i + 1, j + 1] == 1:
                        lambd1[i + 1, j + 1] = evals[k]

    # https://math.stackexchange.com/questions/1119668/determine-a-matrix-knowing-its-eigenvalues-and-eigenvectors/1119690
    for i in range(h):
        for j in range(w):
            if lambd1[i, j] != 1:#True
                # S = [evecs1, evecs2]== [[-df/dy, -df/dx],[df/dx, -df/dy]]
                S = np.array([[1, -0.003 * (i - ori_coord_h) ** 2], [0.003 * (i - ori_coord_h) ** 2, 1]])  # tensor1 S is orthogonal matrix
                # S = np.array([[1, -0.0004 * (i - ori_coord_h) ** 3], [0.0004 * (i - ori_coord_h) ** 3, 1]]) # tensor2
                # S = np.array([[1,-30*np.cos(0.0005*i**2)*0.0016*i],[30*np.cos(0.0005*i**2)*0.001*i,1]]) # tensor3
                # M = diag(evals1,evals2)
                M = np.array([[lambd1[i, j], 0], [0, lambd2[i, j]]])
                S_inv = np.linalg.inv(S)
                T = np.dot(np.dot(S, M), S_inv)
                tensor[0, i, j] = T[0, 0]
                tensor[1, i, j] = T[0, 1]
                tensor[2, i, j] = T[1, 1]
                # sphere metric
                # tensor[0, i, j] = np.sin(j/100*np.pi)
                # tensor[1, i, j] = 0
                # tensor[2, i, j] = 1
            else:
                tensor[0, i, j] = 1
                tensor[1, i, j] = 0
                tensor[2, i, j] = 1

            if i not in range(bezel, h - bezel):
                tensor[0, i, j] = 1
                tensor[1, i, j] = 0
                tensor[2, i, j] = 1

            if j not in range(bezel, w - bezel):
                tensor[0, i, j] = 1
                tensor[1, i, j] = 0
                tensor[2, i, j] = 1

    vis_ellipses(tensor)
    sio.savemat('tensor1.mat', {'tensor': tensor})
