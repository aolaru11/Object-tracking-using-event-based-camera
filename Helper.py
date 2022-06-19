import math

from Visualization import draw_bounding_box_tracking
from particle_filter import ParticleFilter


def preserve_cluster_order(current, object_dictionary):
    result = {}
    for i, centroid_current in enumerate(current):
        min = 10000
        for j, (centroid_prev, particles) in object_dictionary.items():
            distance: float = math.sqrt((centroid_current[0] - centroid_prev[0]) * (centroid_current[0] - centroid_prev[0]) +
                                 (centroid_current[1] - centroid_prev[1]) * (centroid_current[1] - centroid_prev[1]))
            if distance < min:
                min_to_update = (centroid_current, particles)
                min = distance
                aux = j
        if min != 10000:
            result[aux] = min_to_update

    return result


def objects_update(cluster_centers, objects, particle_number):
    objects = preserve_cluster_order(cluster_centers, objects)
    if len(objects) != len(cluster_centers):
        list_cl = []
        for key, (cen, part) in objects.items():
            list_cl.append(cen)
        for cluster in cluster_centers:
            if (cluster == list_cl).all(1).any():
                x = cluster[1]
                y = cluster[0]
                objects[len(objects)] = (cluster, ParticleFilter(particle_number, 34, 34, len(objects), x, y))

    return objects


def perform_particle_filter(objects, number_segments, time, frame, bounding_box_width, bounding_box_height):
    corrected = []
    for count, (centroid, particle) in objects.items():
        x_center, y_center = centroid[1], centroid[0]
        particle.predict(number_segments)
        particle.update_one_point(x_center, y_center)

        if particle.return_effective_N() < particle.N / 2:
            particle.new_resample()

        x_estimated, y_estimated = particle.estimate()
        corrected.append((x_estimated, y_estimated))

        draw_bounding_box_tracking(x_estimated, y_estimated, frame, bounding_box_width, bounding_box_height)

    return corrected
