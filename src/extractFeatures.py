import cv2
import pickle, os
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

def extract_sift_features(X):
    if X.dtype != np.uint8:
        X = X.astype(np.uint8)
    X = np.expand_dims(X,axis=-1)
    image_descriptors = []
    sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.SIFT_create()

    for i in range(len(X)):
        kp, des = sift.detectAndCompute(X[i], None)
        image_descriptors.append(des)

    return image_descriptors


def kmeans_bow(all_descriptors, num_clusters):
    bow_dict = []

    # kmeans = KMeans(n_clusters=num_clusters,init='random',verbose=False).fit(all_descriptors)
    print("using minikmean")
    kmeans = MiniBatchKMeans(n_clusters=5,
                             random_state=0,
                             batch_size=512,
                             max_iter=1000).fit(all_descriptors)
    bow_dict = kmeans.cluster_centers_

    return bow_dict



def create_features_bow(image_descriptors, BoW, num_clusters):
    X_features = []

    for i in range(len(image_descriptors)):
        features = np.array([0] * num_clusters)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)
            argmin = np.argmin(distance, axis=1)
            
            for j in argmin:
                features[j] += 1

        X_features.append(features)

    return X_features

def create_BoWSIFTfeature(X_descriptors,num_clusters):
    if os.path.isfile('BoW_data/bow_dict_{}.pkl'.format(num_clusters)):
        print("Done load available result")
        return pickle.load(open('BoW_data/bow_dict_{}.pkl'.format(num_clusters), 'rb'))
    print("Processing...")
    all_descriptors = []
    for descriptors in X_descriptors:
        if descriptors is not None:
            for des in descriptors:
                all_descriptors.append(des)
    print('Total number of descriptors: %d' %(len(all_descriptors)))
    bow_dict = kmeans_bow(all_descriptors,num_clusters)
    pickle.dump(bow_dict, open('BoW_data/bow_dict_{}.pkl'.format(num_clusters), 'wb'))
    print('Data saved at BoW_data/bow_dict_{}.pkl'.format(num_clusters))
    return bow_dict