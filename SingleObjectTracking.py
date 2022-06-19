import os
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
import math

from sklearn.cluster import DBSCAN

from Evaluation import check_contour, check_order, accuracy_tracking, tracking_threshold, \
    compare_centroids_ground_truth_result
from Helper import objects_update, perform_particle_filter
from Visualization import show_resized_image
from particle_filter import ParticleFilter
import eventvision as ev



bounding_box_width = 9
bounding_box_height = 9


def compute_center_rectangle(x_min, y_min, x_max, y_max):
    """
    Compute center of a rectangle
    :param x_min: x coordinate rectangle
    :param y_min: y coordinate rectangle
    :param x_max: x coordinate rectangle
    :param y_max: y coordinate rectangle
    :return: center
    """
    x_center = (x_max + x_min) // 2
    y_center = (y_max + y_min) // 2
    return x_center, y_center


def find_rectangle_borders(events):
    """
    Estimate rectangle around object
    :param events: list of events
    :return: corners coordinate rectangle
    """
    x_min = 1000
    y_min = 1000
    x_max = -1
    y_max = -1
    for x, y, t, p in events:
        if x > x_max and y > y_max:
            x_max = x
            y_max = y
        if x < x_min and y < y_min:
            x_min = x
            y_min = y

    return math.floor(x_min), math.floor(y_min), math.floor(x_max), math.floor(y_max)


def rectangle_around_single_cluster(selected_events, clustering_result, value):
    """
    Get events from cluster
    :param selected_events: list of events
    :param clustering_result: labels cluster
    :param value: specific label
    :return: events within a cluster
    """
    actual_events = selected_events[clustering_result == value]
    return find_rectangle_borders(actual_events)


def construct_frame(TD, frame_start, frame_end):
    """
     Construct frame
    :param TD: events
    :param frame_start: start time of frame
    :param frame_end: end time of frame
    :return: frame
    """
    current_frame = TD.data[(TD.data.ts >= frame_start) & (TD.data.ts < frame_end)]
    td_img = np.ones((34, 34), dtype=np.uint8)

    # visualize frame
    if current_frame.size > 0:
        td_img.fill(128)
        for datum in np.nditer(current_frame):
            td_img[datum['y'].item(0), datum['x'].item(0)] = datum['p'].item(0)

        td_img = np.piecewise(td_img, [td_img == 0, td_img == 1, td_img == 128], [0, 255, 128])

    return td_img


def init_objects(dict_segments, particle_number):
    """
    Initialize the objects
    :param dict_segments: list of events divided on time frames
    :param particle_number: number of particles
    :return: dictionary with centroids and particle sets, set of centroids
    """
    if len(dict_segments) == 0:
        return None

    objects = {}
    events = dict_segments[0]
    epsilon = 15

    clustering = DBSCAN(eps=epsilon).fit_predict(events)
    cluster_centers = []
    for label in set(clustering):
        x_min, y_min, x_max, y_max = rectangle_around_single_cluster(events, clustering, label)
        cluster_centers.append(compute_center_rectangle(x_min, y_min, x_max, y_max))

    for count, centroid in enumerate(cluster_centers):
        x = centroid[1]
        y = centroid[0]
        objects[count] = (centroid, ParticleFilter(particle_number, 34, 34, count, x, y))

    return objects, cluster_centers


def separate_events_time(frame_length, td):
    """
      Divide events within time frames
      :param frame_length: time frame length
      :param td: events
      :return: list of events
    """
    dict_segments = {}
    t_max = td.data.ts[-1]
    frame_start = td.data[0].ts
    frame_end = td.data[0].ts + frame_length
    count = 0
    while frame_start < t_max:
        frame_data = td.data[(td.data.ts >= frame_start) & (td.data.ts < frame_end)]

        if frame_data.size > 0:
            events = []
            for i in range(len(frame_data)):
                events.append([frame_data.x[i], frame_data.y[i], frame_data.ts[i] * 0.0001, frame_data.p[i] * 0])
            events = np.asarray(events)
            dict_segments[count] = events
        count += 1

        frame_start = frame_end + 1
        frame_end = frame_end + frame_length + 1

    return dict_segments



def read_data(className, index):
    """
    Read a specific file from a folder using the index of the file
    :param className: folder name
    :param index: index of file
    :return: data
    """
    onlyFiles = [f for f in listdir(className) if isfile(join(className, f))]
    current_file = onlyFiles[index]
    TD = ev.read_dataset(os.path.join(className, current_file))
    return TD


def read_data_file(file_name):
    """
    Read data directly from the specified file
    :param file_name: file name
    :return: data
    """
    TD = ev.read_dataset(file_name)
    return TD


def single_object_tracking(file_name,  particle_number, frame_length):
    """
       Perform single target tracking
       :param file_name: file with events
       :param particle_number: number of particles
       :param frame_length: time frame length in microseconds
       :return:
    """
    epsilon = 15
    # td = read_data(filename, 2)
    td = read_data_file(file_name)
    dict_segments = separate_events_time(frame_length, td)
    number_segments = len(dict_segments)
    result = []
    objects, prev = init_objects(dict_segments, particle_number)
    tracking_accuracy = []
    for i in range(0, len(dict_segments)):
        print('Time frame number: ' + str(i))

        selected_events = dict_segments[i]
        frame = construct_frame(td, selected_events[0][2] / 0.0001, selected_events[len(selected_events) - 1][2] / 0.0001)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        time = selected_events[len(selected_events) - 1][2] / 0.0001 - selected_events[0][2] / 0.0001
        time /= 1000

        # perform clustering
        clustering = DBSCAN(eps=epsilon).fit_predict(selected_events)
        cluster_centers = []

        # compute centroids
        for label in set(clustering):
            x_min, y_min, x_max, y_max = rectangle_around_single_cluster(selected_events, clustering, label)
            cluster_centers.append(compute_center_rectangle(x_min, y_min, x_max, y_max))

        # maintain order
        objects_update(cluster_centers, objects, particle_number)

        corrected = perform_particle_filter(objects, number_segments, time, frame, bounding_box_width, bounding_box_height)

        # compute the distance between the estimated centroid and the ground truth
        annotation = 'ground truth/image_single' + str(i) +'.png'
        tracking_accuracy += compare_centroids_ground_truth_result(annotation, corrected)


        # show image
        show_resized_image(frame)
        result.append(frame)

    accuracy = accuracy_tracking(tracking_accuracy)
    print('Tracking accuracy: ' + str(accuracy) + ' with tracking distance error of ' + str(tracking_threshold))


if __name__ == '__main__':
    file_name = '2_class/00309.bin'
    number_of_particles = 100
    frame_length = 24e3

    single_object_tracking(file_name, number_of_particles, frame_length)


