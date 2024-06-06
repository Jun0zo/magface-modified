import numpy as np
import cv2
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
import math

sns.set_theme(style="white")


def imshow(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def main(feat_list_file):
    with open(feat_list_file, 'r') as f:
        lines = f.readlines()

    img_2_feats = {}
    img_2_mag = {}
    img_2_nidx = {}

    nidx_2_mag = {}
    for line in lines:
        parts = line.strip().split(' ')
        imgname = parts[0]
        feats = [float(e) for e in parts[1:]]
        mag = np.linalg.norm(feats)
        img_2_feats[imgname] = feats / mag
        img_2_mag[imgname] = mag
        img_2_nidx[imgname] = int(imgname.split(
            '/')[-1].split('.')[0].split('_')[0])

        nidx_2_mag[img_2_nidx[imgname]] = mag

    imgnames = list(img_2_mag.keys())
    # sort names
    for imgname in imgnames:
        # print('imgname :', imgname.split('/')[-1])
        pass

    mags = [img_2_mag[imgname] for imgname in imgnames]
    # print('mags : ', mags)
    sort_idx = np.argsort(mags)

    H, W = 224, 224
    # n x m

    num_images = len(sort_idx)
    NH = int(math.sqrt(num_images))
    NW = math.ceil(num_images / NH)

    canvas = np.zeros((NH * H, NW * W, 3), np.uint8)

    for i, ele in enumerate(sort_idx):
        img_full_path = imgnames[ele]
        imgname = '/'.join(imgnames[ele].split('/')[-2:])
        img = cv2.imread(img_full_path)
        img = cv2.resize(img, (W, H))
        y_offset = int(i / NW) * H
        x_offset = (i % NW) * W
        canvas[y_offset: y_offset + H, x_offset: x_offset + W, :] = img

    plt.figure(figsize=(10, 10))
    # print([float('{0:.2f}'.format(mags[idx_])) for idx_ in sort_idx])
    imshow(canvas)

    feats = np.array([img_2_feats[imgnames[ele]] for ele in sort_idx])
    sim_mat = np.dot(feats, feats.T)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.heatmap(sim_mat, cmap="PuRd", annot=True)

    nidx = [img_2_nidx[imgname] for imgname in imgnames]
    sort_nidx = np.argsort(nidx)
    print('sort_nidx : ', sort_nidx)
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.bar(range(len(sort_nidx)), [
            nidx_2_mag[nidx[idx_]] for idx_ in sort_nidx])

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Visualization')
    parser.add_argument('--feat_file', type=str, default='toy_imgs/feat.list')
    args = parser.parse_args()

    main(args.feat_file)
