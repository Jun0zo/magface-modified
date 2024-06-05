import numpy as np
import cv2
from matplotlib import pyplot as plt
import seaborn as sns
import argparse

sns.set_theme(style="white")


def imshow(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# def get_feature()


def main(feat_list_file):
    with open(feat_list_file, 'r') as f:
        lines = f.readlines()

    img_2_feats = {}
    img_2_mag = {}
    for line in lines:
        parts = line.strip().split(' ')
        imgname = parts[0]
        feats = [float(e) for e in parts[1:]]
        mag = np.linalg.norm(feats)
        img_2_feats[imgname] = feats/mag
        img_2_mag[imgname] = mag

    imgnames = list(img_2_mag.keys())
    mags = [img_2_mag[imgname] for imgname in imgnames]

    sort_idx = np.argsort(mags)

    H, W = 224, 224
    NH, NW = 1, len(sort_idx)
    canvas = np.zeros((NH * H, NW * W, 3), np.uint8)

    for i, ele in enumerate(sort_idx):
        # print('image name :', imgnames[ele])

        img_full_path = imgnames[ele]

        imgname = imgnames[ele].split('/')[-1].split('.')[0]

        img = cv2.imread(img_full_path)
        img = cv2.resize(img, (W, H))
        canvas[int(i / NW) * H: (int(i / NW) + 1) * H, (i %
                                                        NW) * W: ((i % NW) + 1) * W, :] = img

    # dimension reduction (EM-cluster)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)

    img_full_path = imgnames[ele]
    img_names = [img_full_path.split('/')[-1].split('.')[0]
                 for img_full_path in imgnames]

    feats = np.array([img_2_feats[imgnames[ele]] for ele in sort_idx])

    names = [img_names[ele].split('_')[0] for ele in sort_idx]
    resolutions = [int(img_names[ele].split('_')[-1]) for ele in sort_idx]
    mags = [float('{0:.2f}'.format(mags[idx_])) for idx_ in sort_idx]

    pca_feats = pca.fit_transform(feats)
    print(resolutions)

    # plot features (img_2_feats)
    plt.figure(figsize=(5, 5))
    plt.scatter(pca_feats[:, 0], pca_feats[:, 1],
                c=resolutions, cmap='viridis')
    for i, txt in enumerate(names):
        plt.annotate(f'{names[i]}_{mags[i]}_{resolutions[i]}',
                     (pca_feats[i, 0], pca_feats[i, 1]))
    plt.show()

    # plt.show()


# python inf.py --feat_file sample_data/exp1/feat.list  : 뒤로 점점 멀어지는 실험
# python inf.py --feat_file sample_data/exp2/feat.list  : 다양한 각도에서 찍는 실험
# feat_file (sample_data/exp1/feat.list)에 있는 절대경로를 수정해야 함!!
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Visualization')
    parser.add_argument('--feat_file', type=str, default='toy_imgs/feat.list')
    args = parser.parse_args()

    main(args.feat_file)
