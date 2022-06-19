import cv2
from sklearn.cluster import estimate_bandwidth, MeanShift
import numpy as np
from sklearn.ensemble import IsolationForest

from Evaluation import compare_centroids_ground_truth_result, accuracy_tracking, tracking_threshold
from Helper import objects_update, perform_particle_filter
from Visualization import show_resized_image
from particle_filter import ParticleFilter

bounding_box_width = 9
bounding_box_height = 9


def construct_frame_from_npy(data, frame_start, frame_end):
    """
    Construct frame
    :param data: events
    :param frame_start: start time of frame
    :param frame_end: end time of frame
    :return: frame
    """
    height = -1
    width = -1
    for (y, x, t, p) in data:
        if x > width:
            width = x
        if y > height:
            height = y

    td_img = np.ones((height + 1, width + 1), dtype=np.uint8)

    index = []
    for i in range(data.shape[0]):
        if (data[i][2] >= frame_start) and (data[i][2] < frame_end):
            index.append(i)

    current_frame = data[index]

    # visualize frame
    if current_frame.size > 0:
        td_img.fill(128)

        for datum in current_frame:
            td_img[datum[0], datum[1]] = datum[3]
        td_img = np.piecewise(td_img, [td_img == 0, td_img == 1, td_img == 128], [0, 255, 128])

    return td_img


def remove_anomalies(events):
    """
    Remove anomalies
    :param events: list of events
    :return: List of events without noise
    """
    cleaned_events = IsolationForest(random_state=0, n_jobs=-1, contamination=0.05).fit(events)
    unwanted_events = cleaned_events.predict(events)
    selected_events_cleaned = events[np.where(unwanted_events == 1, True, False)]
    return selected_events_cleaned


def show_multiple_digits_frame(frame_length, data):
    """
    Divide events within time frames
    :param frame_length: time frame length
    :param data: events
    :return: list of events
    """
    height = -1
    width = -1
    for (y, x, t, p) in data:
        if x > width:
            width = x
        if y > height:
            height = y

    t_max = data[-1][2]
    frame_start = data[0][2]
    frame_end = data[0][2] + frame_length
    dict_events = []
    td_img = np.ones((height + 1, width + 1), dtype=np.uint8)
    while frame_start < t_max:
        index = []
        for i in range(data.shape[0]):
            if (data[i][2] >= frame_start) and (data[i][2] < frame_end):
                index.append(i)

        frame_data = data[index]
        if frame_data.size > 0:
            td_img.fill(128)
            events = []

            for datum in frame_data:
                td_img[datum[0], datum[1]] = datum[3]
                events.append([datum[0], datum[1], datum[2] * 0.0001, datum[3]])
            events = np.asarray(events)

            # to remove anomalies
            events = remove_anomalies(events)

            dict_events.append(events)
            td_img = np.piecewise(td_img, [td_img == 0, td_img == 1, td_img == 128], [0, 255, 128])

        frame_start = frame_end + 1
        frame_end = frame_end + frame_length + 1

    return dict_events


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
    BW = estimate_bandwidth(events)
    clustering = MeanShift(bandwidth=BW).fit(events)
    cluster_centers = clustering.cluster_centers_
    for count, centroid in enumerate(cluster_centers):
        x = centroid[1]
        y = centroid[0]
        objects[count] = (centroid, ParticleFilter(particle_number, 34, 34, count, x, y))

    return objects, cluster_centers


def multi_targets_tracking(file_name, particle_number, frame_length):
    """
    Perform multi target tracking
    :param file_name: file with events
    :param particle_number: number of particles
    :param frame_length: time frame length in microseconds
    :return:
    """
    data = np.load(file_name)
    dict_segments = show_multiple_digits_frame(frame_length, data)
    number_segments = len(dict_segments)
    result = []
    objects, prev = init_objects(dict_segments, particle_number)
    tracking_accuracy = []
    for i in range(0, len(dict_segments)):
        print('Time frame number: ' + str(i))

        selected_events = dict_segments[i]
        frame = construct_frame_from_npy(data, selected_events[0][2] / 0.0001, selected_events[len(selected_events) - 1][2] / 0.0001)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        time = selected_events[len(selected_events) - 1][2] / 0.0001 - selected_events[0][2] / 0.0001
        time /= 1000
        BW = estimate_bandwidth(selected_events)
        clustering = MeanShift(bandwidth=BW).fit(selected_events)
        cluster_centers = clustering.cluster_centers_

        objects_update(cluster_centers, objects, particle_number)

        corrected = perform_particle_filter(objects, number_segments, time, frame, bounding_box_width, bounding_box_height)

        annotation = 'ground truth/img' + str(i) +'.png'
        tracking_accuracy += compare_centroids_ground_truth_result(annotation, corrected)

        show_resized_image(frame)
        result.append(frame)

    accuracy = accuracy_tracking(tracking_accuracy)
    print('Tracking accuracy: ' + str(accuracy) + ' with tracking distance error of ' + str(tracking_threshold))

if __name__ == '__main__':
    file_name = '9-7-8.npy'
    number_of_particles = 100
    frame_length = 24e3

    multi_targets_tracking (file_name, number_of_particles, frame_length)
