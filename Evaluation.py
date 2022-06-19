import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, estimate_bandwidth, MeanShift
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

tracking_threshold = 4.5


def accuracy_tracking(tracking_accuracy):
    correct = 0
    for tracking_error in tracking_accuracy:
        if tracking_error <= tracking_threshold:
            correct += 1

    acc = correct / len(tracking_accuracy)
    return acc


def check_order(centroids, centers_annotations):
    result = []
    for i, centroid_current in  enumerate(centroids):
        min_dist = 10000
        for j, center_annotations in enumerate(centers_annotations):
            distance = math.sqrt((centroid_current[0] - center_annotations[0]) * (centroid_current[0] - center_annotations[0]) +
                                 (centroid_current[1] - center_annotations[1]) * (centroid_current[1] - center_annotations[1]))
            if distance < min_dist:
                min_dist = distance

        result.append(min_dist)
    return result


def check_contour(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edged = cv2.Canny(image, 30, 200)

    coordinates = []
    contours, hierarchy = cv2.findContours(edged,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ct = 0
    for i, contour in enumerate(contours):
        # Find bounding rectangles
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the rectangle
        if w != 1 and h != 1:
            if ct % 2 == 0:
                x_center = (x + x + w) // 2
                y_center = (y + y + h) // 2
                coordinates.append((x_center, y_center))
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 1)
                cv2.putText(image, str(i), (x, y), 1, 1, [0, 255, 0])
            ct += 1

    return coordinates


def compute_histograms(vx, vy, v):
    histx, binsx = np.histogram(vx)
    plt.hist(vx, binsx)
    plt.title("Velocity on x axis")
    plt.show()
    midsx = 0.5 * (binsx[1:] + binsx[:-1])
    meanx = np.average(midsx, weights=histx)
    print(meanx)

    histy, binsy = np.histogram(vy)
    plt.hist(vy, binsy)
    plt.title("Velocity on y axis")
    plt.show()
    midsy = 0.5 * (binsy[1:] + binsy[:-1])
    meany = np.average(midsy, weights=histy)
    print(meany)

    hist, bins = np.histogram(v)
    plt.hist(v, bins)
    plt.title("Velocity")
    plt.show()
    mids = 0.5 * (bins[1:] + bins[:-1])
    mean = np.average(mids, weights=hist)
    print(mean)


def compute_velocities():
    velocity_x = []
    velocity_y = []
    particle_number = 100
    data = np.load('9-7-8.npy')
    dict_segments = show_multiple_digits_frame(data)
    objects, prev = init_objects(dict_segments, particle_number)
    prev_obj = objects.copy()
    velocity = []
    for i in range(1, len(dict_segments)):
        selected_events = dict_segments[i]
        time = selected_events[len(selected_events) - 1][2] / 0.0001 - selected_events[0][2] / 0.0001
        time /= 1000
        BW = estimate_bandwidth(selected_events)
        clustering = MeanShift(bandwidth=BW).fit(selected_events)
        cluster_centers = clustering.cluster_centers_
        objects = preserve_cluster_order(cluster_centers, objects)
        vx, vy, v = compare_consecutive_frames(prev_obj, objects, time)
        velocity_x += vx
        velocity_y += vy
        velocity += v
        prev_obj = objects.copy()

    compute_histograms(velocity_x, velocity_y, velocity)


def compare_consecutive_frames(centroids1, centroids2, time):
    results_x = []
    results_y = []
    results = []
    for key, (centroid1, p1) in centroids1.items():
        if key not in centroids2:
            continue
        else:
            centroid2, p2 = centroids2[key]
            dist_y = math.fabs(centroid1[0] - centroid2[0])
            dist_x = math.fabs(centroid1[1] - centroid2[1])
            distance = math.sqrt(dist_x * dist_x + dist_y * dist_y)
            results_x.append(float(dist_x / float(time)))
            results_y.append(float(dist_y / float(time)))
            results.append(distance)

    return results_x, results_y, results


def evaluate_clusters(dict_events, epsilon):
    score_calinski_harabasz_score_ms = []
    score_davies_bouldin_score_ms = []
    score_silhoutte_ms = []
    score_calinski_harabasz_score_dbscan = []
    score_davies_bouldin_score_dbscan = []
    score_silhoutte_dbscan = []

    for i in range(0, len(dict_events)):
        selected_events = dict_events[i]
        clustering = DBSCAN(eps=epsilon).fit_predict(selected_events)
        score_silhoutte_dbscan.append(silhouette_score(selected_events, clustering))
        score_calinski_harabasz_score_dbscan.append(calinski_harabasz_score(selected_events, clustering))
        score_davies_bouldin_score_dbscan.append(davies_bouldin_score(selected_events, clustering))

        BW = estimate_bandwidth(selected_events)
        clustering = MeanShift(bandwidth=BW).fit(selected_events)
        score_silhoutte_ms.append(silhouette_score(selected_events, clustering.labels_))
        score_calinski_harabasz_score_ms.append(calinski_harabasz_score(selected_events, clustering.labels_))
        score_davies_bouldin_score_ms.append(davies_bouldin_score(selected_events, clustering.labels_))


def compare_centroids_ground_truth_result(file_name, corrected):
    img = cv2.imread(file_name)
    centroid_annotation = check_contour(img)
    tracking_accuracy = check_order(corrected, centroid_annotation)
    return tracking_accuracy


