from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import os
import numpy as np
from utils.files_op import Files
from tensorflow.keras.preprocessing.image import load_img
from pathlib import Path

ROOT_DIR = Path('.')
ANALYSIS_DIR = os.path.join(ROOT_DIR, 'analysis')


class ClusterDf:

    def __init__(self, df, n_clusters, files, facial):


        self.facial = facial
        kmeans = KMeans(n_clusters, n_init=10, random_state=22)
        kmeans.fit(df)

        self.groups = {}

        for file, cluster in zip(files, kmeans.labels_):
            if cluster not in self.groups.keys():
                self.groups[cluster] = []
                self.groups[cluster].append(file)
            else:
                self.groups[cluster].append(file)


        self.n_clusters = n_clusters
        self.files = files
        self.df = df
        self.kmeans = kmeans



    def lips_size(self, axe0, axe1):

        centroids = self.kmeans.cluster_centers_

        # print(centroids)
        cent1 = 0 if axe0 == axe1 else 1
        plt.scatter(self.df[axe0], self.df[axe1], c=self.kmeans.labels_.astype(float), s=50, alpha=0.5)
        plt.scatter(centroids[:, 0], centroids[:, cent1], c='red', s=50)
        plt.set_xlabel(f'{axe0}', fontsize=15)
        plt.set_ylabel(f'{axe1}', fontsize=15)
        plt.show()


    def view_cluster(self, version, save=True):
        cluster_plot = []

        for i in range(len(self.groups.keys())):
            sub_plot = plt.figure(figsize=(25, 25))

            # Gets the list of filenames for a cluster.
            files_g = self.groups[i]

            # Only allow up to 30 images to be shown at a time.
            if len(files_g) > 30:
                print(f"Clipping cluster size from {len(files_g)} to 30")
                files = files_g[:29]
            # plot each image in the cluster
            for index, file in enumerate(files_g):
                print(file)
                plt.subplot(10, 10, index + 1)

                img = load_img(file)
                img = np.array(img)
                plt.imshow(img)
                plt.axis('off')
            plt.title(f'cluster ID: {i:02d}')
            cluster_plot.append(sub_plot)


        # samples = os.path.join(SAMPLES_DIR, f'samples.jpg')
        for i in range(len(cluster_plot)):
            cluster_fig = cluster_plot[i]

            if save:
                # Save: Cluster plot
                SAMPLES_DIR = os.path.join(ANALYSIS_DIR, f'{self.facial}_{version}')
                if not os.path.exists(SAMPLES_DIR):
                    os.makedirs(SAMPLES_DIR)
                samples = os.path.join(SAMPLES_DIR, f'{self.facial}_{version}_{i:02d}.jpg')
                plt.savefig(samples, bbox_inches='tight', pad_inches=0)

            # plt.show()
            plt.close()
        plt.close()


