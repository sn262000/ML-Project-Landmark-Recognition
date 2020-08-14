import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
from six import BytesIO

from scipy.spatial import cKDTree

def image_input_fn(image_files):
    filename_queue = tf.train.string_input_producer(
        image_files, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)
    return tf.image.convert_image_dtype(image_tf, tf.float32)

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.FATAL)

m = hub.Module('https://tfhub.dev/google/delf/1')

def get_features(image_array):
    # The module operates on a single image at a time, so define a placeholder to
    # feed an arbitrary image in.
    image_placeholder = tf.placeholder(
        tf.float32, shape=(None, None, 3), name='input_image')

    module_inputs = {
        'image': image_placeholder,
        'score_threshold': 100.0,
        'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
        'max_feature_num': 1000,
    }

    module_outputs = m(module_inputs, as_dict=True)

    image_tf = image_input_fn(image_array)

    with tf.train.MonitoredSession() as sess:
        results_dict = {}  # Stores the locations and their descriptors for each image
        for image_path in image_array:
            image = sess.run(image_tf)
#             print('Extracting locations and descriptors from %s' % image_path)
            results_dict[image_path] = sess.run(
                [module_outputs['locations'], module_outputs['descriptors']],
                feed_dict={image_placeholder: image})
    return results_dict

def match_images(results_dict, image_1_path, image_2_path, ax):
    distance_threshold = 0.8

    # Read features.
    locations_1, descriptors_1 = results_dict[image_1_path]
    num_features_1 = locations_1.shape[0]
#     print("Loaded image 1's %d features" % num_features_1)
    locations_2, descriptors_2 = results_dict[image_2_path]
    num_features_2 = locations_2.shape[0]
#     print("Loaded image 2's %d features" % num_features_2)

    # Find nearest-neighbor matches using a KD tree.
    d1_tree = cKDTree(descriptors_1)
    _, indices = d1_tree.query(
        descriptors_2, distance_upper_bound=distance_threshold)

  # Select feature locations for putative matches.
    locations_2_to_use = np.array([
        locations_2[i,]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        locations_1[indices[i],]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])

  # Perform geometric verification using RANSAC.
    _, inliers = ransac(
        (locations_1_to_use, locations_2_to_use),
        AffineTransform,
        min_samples=3,
        residual_threshold=20,
        max_trials=1000)
    # the number of inliers as the score for retrieved images.
#     print('Found %d inliers' % sum(inliers))

    # Visualize correspondences.
#     _, ax = plt.subplots()
    img_1 = mpimg.imread(image_1_path)
    img_2 = mpimg.imread(image_2_path)
    inlier_idxs = np.nonzero(inliers)[0]
    plot_matches(
        ax,
        img_1,
        img_2,
        locations_1_to_use,
        locations_2_to_use,
        np.column_stack((inlier_idxs, inlier_idxs)),
        matches_color='b')
    ax.axis('off')
    ax.set_title('DELF correspondences')
#     plt.show()

def get_delf_features_inliners_coordinates(results_dict, image_1_path, image_2_path):
    distance_threshold = 0.8

    # Read features.
    locations_1, descriptors_1 = results_dict[image_1_path]
    num_features_1 = locations_1.shape[0]
#     print("Loaded image 1's %d features" % num_features_1)
    locations_2, descriptors_2 = results_dict[image_2_path]
    num_features_2 = locations_2.shape[0]
#     print("Loaded image 2's %d features" % num_features_2)

    # Find nearest-neighbor matches using a KD tree.
    d1_tree = cKDTree(descriptors_1)
    _, indices = d1_tree.query(
        descriptors_2, distance_upper_bound=distance_threshold)

  # Select feature locations for putative matches.
    locations_2_to_use = np.array([
        locations_2[i,]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        locations_1[indices[i],]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])

  # Perform geometric verification using RANSAC.
    _, inliers = ransac(
        (locations_1_to_use, locations_2_to_use),
        AffineTransform,
        min_samples=3,
        residual_threshold=20,
        max_trials=1000)
    
    inlier_idxs = np.nonzero(inliers)[0]
#     print(inlier_idxs)
    inliner_x_locations = []
    inliner_y_locations = []
    for inlier_idx in inlier_idxs:
        inliner_x_locations.append(int(locations_1_to_use[inlier_idx][1]))
        inliner_y_locations.append(int(locations_1_to_use[inlier_idx][0]))

    min_x = min(inliner_x_locations)
    max_x = max(inliner_x_locations)
    min_y = min(inliner_y_locations)
    max_y = max(inliner_y_locations)
    return min_x, max_x, min_y, max_y