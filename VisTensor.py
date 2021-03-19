import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
# from euler import eulerpath
# from geodesic import geodesicpath
# from dijkstra import shortpath
from scipy.io import loadmat
from scipy.io import savemat


def is_pd(K):
    try:
        np.linalg.cholesky(K)
        return 1
    except np.linalg.linalg.LinAlgError as err:
        # if 'Matrix is not positive definite' in err.message:
        return 0
        # else:
        #     raise


def vis_ellipses(eps11, eps12, eps22):
    # visualizing ellipses
    ells = []
    tens = np.zeros((2, 2))
    scale = 5e-4
    for x in range(1, eps11.shape[0] - 1):
        for y in range(1, eps11.shape[1] - 1):
            # evals and evecs by numpy
            tens[0, 0] = eps11[x, y]
            tens[0, 1] = eps12[x, y]
            tens[1, 0] = eps12[x, y]
            tens[1, 1] = eps22[x, y]
            evals, evecs = np.linalg.eigh(tens)
            angles = np.degrees(np.math.atan2(evecs[1][0], evecs[1][1]))
            ells.append(Ellipse(xy=(y, x), width=scale * evals[1], height=scale * evals[0], angle=angles))
            # print(evals[1] ** 2 + evals[0] ** 2)
    # plt.figure(1)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(1)
        e.set_facecolor([0, 0, 0])
    ax.set_xlim(0, eps11.shape[1])
    ax.set_ylim(0, eps11.shape[0])
    plt.show()


if __name__ == "__main__":
    # data = loadmat('Data/103818to105923_g1_n_24kL2_10kEbin.mat')['tensor']
    # data = loadmat('Data/105923_orig_tensors_masked_bg.mat')['tensor']
    data = loadmat('Data/105923_scaled_tensors_clip_3.mat')['tensor']
    mask = loadmat('Data/105923_filt_mask.mat')['mask']
    # data = loadmat('Data/105923_scaled_tensors_clip_3_diffeo25_g1.mat')['tensor']
    # data = loadmat('Data/103818_scaled_tensors_clip_3_diffeo25_g1_n.mat')['tensor']
    # eps11 = data[0,0, :, :]
    # eps12 = data[1,0, :, :]
    # eps22 = data[1,1, :, :]
    eps11 = data[0, :, :]
    eps12 = data[1, :, :]
    eps22 = data[2, :, :]
    det = eps11 * eps22 - eps12 * eps12
    weight = mask/det
    # eps11_adjust = eps22 / det
    # eps12_adjust = -eps12 / det
    # eps22_adjust = eps11 / det

    # tens = np.zeros((2, 2))
    # mag = np.zeros(data.shape[-2:])
    # eps11_adjust = eps11.copy()
    # eps12_adjust = eps12.copy()
    # eps22_adjust = eps22.copy()
    # # count = 0
    # for x in range(eps11.shape[0]):
    #     for y in range(eps11.shape[1]):
    #         #         evals and evecs by numpy
    #         # tens[0, 0] = eps11_adjust[x, y]
    #         # tens[0, 1] = eps12_adjust[x, y]
    #         # tens[1, 0] = eps12_adjust[x, y]
    #         # tens[1, 1] = eps22_adjust[x, y]
    #         # evals, evecs = np.linalg.eigh(tens)
    #         # mag[x, y] = evals[1]
    #         if (eps11_adjust[x, y] == eps22_adjust[x, y] and eps12_adjust[x, y] == 0):
    #             eps11_adjust[x, y] = 5e2
    #             eps12_adjust[x, y] = 0
    #             eps22_adjust[x, y] = 5e2
            # if (x == 71 and y == 121) or (x == 69 and y == 116):
            #     eps11_adjust[x, y] = 5e2
            #     eps12_adjust[x, y] = 0
            #     eps22_adjust[x, y] = 5e2
            # mag[x, y] = np.linalg.norm(evals)
    #             count += 1
    #         else:
    #             eps11_adjust[x, y] = eps11[x, y]*5e3
    #             eps12_adjust[x, y] = eps12[x, y]*5e3
    #             eps22_adjust[x, y] = eps22[x, y]*5e3
    #         if (eps11[x, y] == 0 and eps22[x, y] == 0 and eps12[x, y] == 0):
    #             eps11_adjust[x, y] = 0.0005
    #             eps22_adjust[x, y] = 0.0005
    #         if not is_pd([[eps11[x,y],eps12[x,y]],[eps12[x,y],eps22[x,y]]]):
    #             eps11_adjust[x, y] = 0.0005
    #             eps12_adjust[x, y] = 0
    #             eps22_adjust[x, y] = 0.0005
    #         count+=is_pd([[eps11[x,y],eps12[x,y]],[eps12[x,y],eps22[x,y]]])

    # print(count)
    # tensor = np.zeros(data.shape)
    # tensor[0, :, :] = eps11_adjust
    # tensor[1, :, :] = eps12_adjust
    # tensor[2, :, :] = eps22_adjust
    # savemat('Data/105923_orig_tensors_masked_bg_inv_wiped.mat', {'tensor': tensor})
    # vis_ellipses(eps11_adjust, eps12_adjust, eps22_adjust)
    # weight[weight > 1e7] = 0
    plt.imshow(weight)
    plt.show()

    # plt.imshow(det)
    # plt.show()
    #
    # plt.imshow(mask)
    # plt.show()
