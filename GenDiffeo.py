from Packages.GeoPlot import *
from Packages.RegistrationFunc import *
from Packages.SplitEbinMetric import *
import torch
import numpy as np
import scipy.io as sio


def get_idty(size_h, size_w):
    HH, WW = torch.meshgrid([torch.arange(size_h, dtype=torch.double), torch.arange(size_w, dtype=torch.double)])
    return torch.stack((HH, WW))


def divfree_kernel(p1, p2, sigma):
    kernel = np.zeros((2, 2))
    coefficient = 1 / sigma ** 4 * np.exp(-((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) / (2 * sigma ** 2))
    kernel[0, 0] = coefficient * (sigma ** 2 - (p1[1] - p2[1]) ** 2)
    kernel[0, 1] = coefficient * ((p1[0] - p2[0]) * (p1[1] - p2[1]))
    kernel[1, 0] = coefficient * ((p1[0] - p2[0]) * (p1[1] - p2[1]))
    kernel[1, 1] = coefficient * (sigma ** 2 - (p1[0] - p2[0]) ** 2)

    return kernel


def gen_gaussian_diffeo(N=100, step_size=1):
    id = get_idty(N, N).detach().numpy()
    fig1 = plt.figure(num=None, figsize=(5, 5), dpi=100, facecolor='w', edgecolor='k')
    ax = plt.gca()
    ax.set_aspect('equal')
    for h in range(0, id.shape[1], step_size):
        plt.plot(id[0, h, :], id[1, h, :], 'b', linewidth=0.5)
    for w in range(0, id.shape[2], step_size):
        plt.plot(id[0, :, w], id[1, :, w], 'b', linewidth=0.5)

    coords = plt.ginput(2)
    plt.show()
    plt.close(fig1)

    start, end = coords[0], coords[1]
    sigma = 200
    mag = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2
    XX, YY = torch.meshgrid([torch.arange(N, dtype=torch.double), torch.arange(N, dtype=torch.double)])
    rsquare = (XX - start[1]) ** 2 + (YY - start[0]) ** 2
    v = torch.exp(-rsquare / (2 * sigma)) * mag

    # fig = plt.figure()
    # im = plt.imshow(v.numpy())
    # fig.colorbar(im)
    # plt.show()

    diffeo = id.copy()

    while (1):
        # update diffeo
        norm = np.sqrt(mag)
        diffeo[0] = diffeo[0] + (end[0] - start[0]) / norm * v.numpy() / 50
        diffeo[1] = diffeo[1] + (end[1] - start[1]) / norm * v.numpy() / 50

        # display
        fig2 = plt.figure(num=None, figsize=(5, 5), dpi=100, facecolor='w', edgecolor='k')
        ax = plt.gca()
        ax.set_aspect('equal')
        for h in range(0, diffeo.shape[1], step_size):
            plt.plot(diffeo[0, h, :], diffeo[1, h, :], 'b', linewidth=0.5)
        for w in range(0, diffeo.shape[2], step_size):
            plt.plot(diffeo[0, :, w], diffeo[1, :, w], 'b', linewidth=0.5)

        # get cursor positions
        coords = plt.ginput(2)
        plt.show()
        plt.close(fig2)

        # generate new displacement
        start, end = coords[0], coords[1]
        sigma = 800
        mag = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2
        XX, YY = torch.meshgrid([torch.arange(N, dtype=torch.double), torch.arange(N, dtype=torch.double)])
        rsquare = (XX - start[1]) ** 2 + (YY - start[0]) ** 2
        v = torch.exp(-rsquare / (2 * sigma)) * mag

        print('Continue or not? (y/n)')
        flag = input()

        if flag == 'n':
            print('Identity number?')
            name = input()
            sio.savemat('diffeo' + name + '.mat', {'diffeo': diffeo})

            fig2 = plt.figure(num=None, figsize=(5, 5), dpi=100, facecolor='w', edgecolor='k')
            ax = plt.gca()
            ax.set_aspect('equal')
            for h in range(0, diffeo.shape[1], step_size):
                plt.plot(diffeo[0, h, :], diffeo[1, h, :], 'b', linewidth=0.5)
            for w in range(0, diffeo.shape[2], step_size):
                plt.plot(diffeo[0, :, w], diffeo[1, :, w], 'b', linewidth=0.5)
            plt.show()
            # plt.savefig('diffeo'+name+'.png')

            break


def gen_divfree_diffeo(height=145, width=174, sigma=5, point_num=1, grid_step_size=1, animation_step_size=0.1,
                       mode='static'):
    id = get_idty(height, width).detach().numpy()

    fig1 = plt.figure(num=None, figsize=(40, 40), dpi=100, facecolor='w', edgecolor='k')
    ax = plt.gca()
    ax.set_aspect('equal')
    for h in range(0, id.shape[1], grid_step_size):
        plt.plot(id[1, h, :], id[0, h, :], 'b', linewidth=0.5)
    for w in range(0, id.shape[2], grid_step_size):
        plt.plot(id[1, :, w], id[0, :, w], 'b', linewidth=0.5)
    coords = plt.ginput(2 * point_num)
    plt.close(fig1)
    plt.show()

    d = []
    p = []
    p_prime = []
    for i in range(point_num):
        p.append(np.flip(np.asarray(coords[i * 2])))
        p_prime.append(np.flip(np.asarray(coords[i * 2 + 1])))
        d.append(np.flip(np.asarray(coords[i * 2 + 1]) - np.asarray(coords[i * 2])))
    K = np.zeros((2 * point_num, 2 * point_num))
    for i in range(point_num):
        for j in range(point_num):
            K[i * 2:i * 2 + 2, j * 2:j * 2 + 2] = divfree_kernel(p[i], p[j], sigma)
    D = np.concatenate((d))
    K_inv = np.linalg.inv(K)
    W = K_inv.dot(D.T)

    xrange = np.linspace(0, height - 1, height)
    yrange = np.linspace(0, width - 1, width)
    y, x = np.meshgrid(yrange, xrange)

    coefficient = []
    [u, v] = [0, 0]
    for i in range(point_num):
        coefficient.append(1 / sigma ** 4 * np.exp(-((x - p[i][0]) ** 2 + (y - p[i][1]) ** 2) / (2 * sigma ** 2)))
        u += coefficient[i] * (
                (sigma ** 2 - (y - p[i][1]) ** 2) * W[i * 2] + (x - p[i][0]) * (y - p[i][1]) * W[i * 2 + 1])
        v += coefficient[i] * (
                (x - p[i][0]) * (y - p[i][1]) * W[i * 2] + (sigma ** 2 - (x - p[i][0]) ** 2) * W[i * 2 + 1])
        # print(u[int(p[i][0]),int(p[i][1])],v[int(p[i][0]),int(p[i][1])])

    diffeo = id.copy()

    while (1):
        # update diffeo
        diffeo[0] = diffeo[0] + u
        diffeo[1] = diffeo[1] + v

        # display
        if mode == 'static':
            fig2 = plt.figure(num=1, figsize=(40, 40), dpi=100, facecolor='w', edgecolor='k')
            ax = plt.gca()
            ax.set_aspect('equal')
            for h in range(0, diffeo.shape[1], grid_step_size):
                plt.plot(diffeo[1, h, :], diffeo[0, h, :], 'b', linewidth=0.5)
            for w in range(0, diffeo.shape[2], grid_step_size):
                plt.plot(diffeo[1, :, w], diffeo[0, :, w], 'b', linewidth=0.5)
            coords = plt.ginput(point_num * 2)
            plt.close(fig2)
            plt.show()
        else:
            for i in range(int(1 / animation_step_size)):
                diffeo[0] = diffeo[0] + u * animation_step_size
                diffeo[1] = diffeo[1] + v * animation_step_size
                fig2 = plt.figure(num=1, figsize=(40, 40), dpi=100, facecolor='w', edgecolor='k')
                plt.clf()
                ax = plt.gca()
                ax.set_aspect('equal')
                for h in range(0, diffeo.shape[1], grid_step_size):
                    plt.plot(diffeo[1, h, :], diffeo[0, h, :], 'b', linewidth=0.5)
                for w in range(0, diffeo.shape[2], grid_step_size):
                    plt.plot(diffeo[1, :, w], diffeo[0, :, w], 'b', linewidth=0.5)
                if i != int(1 / animation_step_size) - 1:
                    plt.pause(1)
                else:
                    coords = plt.ginput(point_num * 2)
                    plt.close(fig2)
                    plt.show()

        d = []
        p = []
        p_prime = []
        for i in range(point_num):
            p.append(np.flip(np.asarray(coords[i * 2])))
            p_prime.append(np.flip(np.asarray(coords[i * 2 + 1])))
            d.append(np.flip(np.asarray(coords[i * 2 + 1]) - np.asarray(coords[i * 2])))
        K = np.zeros((2 * point_num, 2 * point_num))
        for i in range(point_num):
            for j in range(point_num):
                K[i * 2:i * 2 + 2, j * 2:j * 2 + 2] = divfree_kernel(p[i], p[j], sigma)
        D = np.concatenate((d))
        K_inv = np.linalg.inv(K)
        W = K_inv.dot(D.T)

        coefficient = []
        [u, v] = [0, 0]
        for i in range(point_num):
            coefficient.append(1 / sigma ** 4 * np.exp(-((x - p[i][0]) ** 2 + (y - p[i][1]) ** 2) / (2 * sigma ** 2)))
            u += coefficient[i] * (
                    (sigma ** 2 - (y - p[i][1]) ** 2) * W[i * 2] + (x - p[i][0]) * (y - p[i][1]) * W[i * 2 + 1])
            v += coefficient[i] * (
                    (x - p[i][0]) * (y - p[i][1]) * W[i * 2] + (sigma ** 2 - (x - p[i][0]) ** 2) * W[i * 2 + 1])
            # print(u[int(p[i][0]),int(p[i][1])],v[int(p[i][0]),int(p[i][1])])

        print('Continue or not? (y/n)')
        flag = input()

        if flag == 'n':
            print('Identity number?')
            name = input()
            sio.savemat('diffeo' + name + '.mat', {'diffeo': diffeo})

            plt.figure(num=1, figsize=(5, 5), dpi=100, facecolor='w', edgecolor='k')
            ax = plt.gca()
            ax.set_aspect('equal')
            for h in range(0, diffeo.shape[1], grid_step_size):
                plt.plot(diffeo[1, h, :], diffeo[0, h, :], 'b', linewidth=0.5)
            for w in range(0, diffeo.shape[2], grid_step_size):
                plt.plot(diffeo[1, :, w], diffeo[0, :, w], 'b', linewidth=0.5)
            plt.show()
            # plt.savefig('diffeo'+name+'.png')
            det_gamma = get_jacobian_determinant(torch.from_numpy(diffeo))
            fig = plt.figure(num=2)
            im = plt.imshow(det_gamma)
            fig.colorbar(im)
            plt.show()

            break


def animate_divfree_diffeo(N=100, sigma=5, point_num=1, grid_step_size=1, animation_step_size=0.1):
    id = get_idty(N, N).detach().numpy()

    fig1 = plt.figure(num=None, figsize=(40, 40), dpi=100, facecolor='w', edgecolor='k')
    ax = plt.gca()
    ax.set_aspect('equal')
    for h in range(0, id.shape[1], grid_step_size):
        plt.plot(id[1, h, :], id[0, h, :], 'b', linewidth=0.5)
    for w in range(0, id.shape[2], grid_step_size):
        plt.plot(id[1, :, w], id[0, :, w], 'b', linewidth=0.5)
    coords = plt.ginput(2 * point_num)
    plt.close(fig1)
    plt.show()

    d = []
    p = []
    p_prime = []
    for i in range(point_num):
        p.append(np.flip(np.asarray(coords[i * 2])))
        p_prime.append(np.flip(np.asarray(coords[i * 2 + 1])))
        d.append(np.flip(np.asarray(coords[i * 2 + 1]) - np.asarray(coords[i * 2])))
        # print(p, p_prime, d)

    K = np.zeros((2 * point_num, 2 * point_num))
    for i in range(point_num):
        for j in range(point_num):
            K[i * 2:i * 2 + 2, j * 2:j * 2 + 2] = divfree_kernel(p[i], p[j], sigma)
    D = np.concatenate((d))
    K_inv = np.linalg.inv(K)
    W = K_inv.dot(D.T)

    xrange = np.linspace(0, N - 1, N)
    y, x = np.meshgrid(xrange, xrange)

    coefficient = []
    [u, v] = [0, 0]
    for i in range(point_num):
        coefficient.append(1 / sigma ** 4 * np.exp(-((x - p[i][0]) ** 2 + (y - p[i][1]) ** 2) / (2 * sigma ** 2)))
        u += coefficient[i] * (
                (sigma ** 2 - (y - p[i][1]) ** 2) * W[i * 2] + (x - p[i][0]) * (y - p[i][1]) * W[i * 2 + 1])
        v += coefficient[i] * (
                (x - p[i][0]) * (y - p[i][1]) * W[i * 2] + (sigma ** 2 - (x - p[i][0]) ** 2) * W[i * 2 + 1])
        # print(u[int(p[i][0]),int(p[i][1])],v[int(p[i][0]),int(p[i][1])])

    diffeo = id.copy()

    while 1:
        # update diffeo
        diffeo[0] = diffeo[0] + u
        diffeo[1] = diffeo[1] + v

        # display
        for i in range(int(1 / animation_step_size)):
            diffeo[0] = diffeo[0] + u * animation_step_size
            diffeo[1] = diffeo[1] + v * animation_step_size
            fig2 = plt.figure(num=1, figsize=(40, 40), dpi=100, facecolor='w', edgecolor='k')
            plt.clf()
            ax = plt.gca()
            ax.set_aspect('equal')
            for h in range(0, diffeo.shape[1], grid_step_size):
                plt.plot(diffeo[1, h, :], diffeo[0, h, :], 'b', linewidth=0.5)
            for w in range(0, diffeo.shape[2], grid_step_size):
                plt.plot(diffeo[1, :, w], diffeo[0, :, w], 'b', linewidth=0.5)
            if i != int(1 / animation_step_size) - 1:
                plt.pause(1)
            else:
                coords = plt.ginput(point_num * 2)
                print(coords)
                plt.close(fig2)
                plt.show()

        d = []
        p = []
        p_prime = []
        for i in range(point_num):
            p.append(np.flip(np.asarray(coords[i * 2])))
            p_prime.append(np.flip(np.asarray(coords[i * 2 + 1])))
            d.append(np.flip(np.asarray(coords[i * 2 + 1]) - np.asarray(coords[i * 2])))
        K = np.zeros((2 * point_num, 2 * point_num))
        for i in range(point_num):
            for j in range(point_num):
                K[i * 2:i * 2 + 2, j * 2:j * 2 + 2] = divfree_kernel(p[i], p[j], sigma)
        D = np.concatenate((d))
        K_inv = np.linalg.inv(K)
        W = K_inv.dot(D.T)

        coefficient = []
        [u, v] = [0, 0]
        for i in range(point_num):
            coefficient.append(1 / sigma ** 4 * np.exp(-((x - p[i][0]) ** 2 + (y - p[i][1]) ** 2) / (2 * sigma ** 2)))
            u += coefficient[i] * (
                    (sigma ** 2 - (y - p[i][1]) ** 2) * W[i * 2] + (x - p[i][0]) * (y - p[i][1]) * W[i * 2 + 1])
            v += coefficient[i] * (
                    (x - p[i][0]) * (y - p[i][1]) * W[i * 2] + (sigma ** 2 - (x - p[i][0]) ** 2) * W[i * 2 + 1])

        print('Continue or not? (y/n)')
        flag = input()

        if flag == 'n':
            print('Identity number?')
            name = input()
            sio.savemat('diffeo' + name + '.mat', {'diffeo': diffeo})

            # fig2 =
            plt.figure(num=1, figsize=(5, 5), dpi=100, facecolor='w', edgecolor='k')
            ax = plt.gca()
            ax.set_aspect('equal')
            for h in range(0, diffeo.shape[1], grid_step_size):
                plt.plot(diffeo[1, h, :], diffeo[0, h, :], 'b', linewidth=0.5)
            for w in range(0, diffeo.shape[2], grid_step_size):
                plt.plot(diffeo[1, :, w], diffeo[0, :, w], 'b', linewidth=0.5)
            plt.show()
            # plt.savefig('diffeo'+name+'.png')
            det_gamma = get_jacobian_determinant(torch.from_numpy(diffeo))
            fig = plt.figure(num=2)
            im = plt.imshow(det_gamma)
            fig.colorbar(im)
            plt.show()

            break


if __name__ == "__main__":
    # gen_gaussian_diffeo()
    gen_divfree_diffeo(145, 174, sigma=15, point_num=1, grid_step_size=1, animation_step_size=0.1, mode='static')
    # animate_divfree_diffeo(N=100, sigma=25, point_num=2, grid_step_size=1, animation_step_size=0.1)
