import math
from filterpy.monte_carlo import systematic_resample

import cv2
import numpy as np


class ParticleFilter():

    def __init__(self, number_particle, width, height, object_id, x_center, y_center):
        """
        Initialize the set of particles using a uniform distribution
        :param number_particle: The number of particles in the set
        :param width: Width of frame
        :param height: Height of frame
        :param object_id: The id of the object which should be linked to the set of particles
        :param x_center: X coordinate of the observation
        :param y_center: Y coordinate of the observation
        """
        self.k_resample = 0
        self.N = number_particle
        self.particles_event = np.empty((self.N, 3))
        self.prev = np.empty((self.N, 3))
        self.width = width
        self.height = height
        range = 15

        #y, x
        self.particles_event[:, 1] = np.random.uniform(x_center - range, x_center + range, self.N)
        self.particles_event[:, 0] = np.random.uniform(y_center - range, y_center + range, self.N)
        self.particles_event[:, 2] = np.random.uniform(0, 1, self.N)
        self.initial = self.particles_event
        self.weight = np.full(self.N, 1.0)
        self.h = 23
        self.f = 8
        self.w = self.h / self.f
        self.g = 4
        self.s = 28
        self.object_id = object_id

    def motion_model(self):
        """
        Compute motion model considering the normal distribution
        :return: an array of shape [n_particles, 3] with the motion model computed from a normal distribution
        """
        delta_distance = self.w / self.g
        delta_rotation = 1 / (self.s * self.g)

        motion_model = np.zeros((self.N, 3))
        normal_distr_distance = np.random.normal(0, delta_distance * delta_distance, self.N)
        normal_distr_rotation = np.random.normal(0, delta_rotation * delta_rotation, self.N)
        motion_model[:, 0] = normal_distr_distance
        motion_model[:, 1] = normal_distr_distance
        motion_model[:, 2] = normal_distr_rotation
        return motion_model

    def predict(self, size):
        """
        Predict new position
        :param size: Number of time windows
        :return:
        """
        self.prev = self.particles_event
        self.particles_event += self.motion_model() * math.sqrt(size) + 0.1

    def distance_difference_one_point(self, x, y):
        """
        Compute distance between particle and the observed point
        :param x: x coordinate observed point
        :param y: y coordinate observed point
        :return: array with distances
        """
        position = np.empty((self.N, 2))
        position[:, 1].fill(x)
        position[:, 0].fill(y)
        distance = np.linalg.norm(self.particles_event[:, 0:2] - position, axis=1)
        max_distance = np.amax(distance)
        distance = np.add(-distance, max_distance)
        return distance

    def update_one_point(self, x, y):
        """
        Update score of particles
        :param x: x coordinate observed point
        :param y: y coordinate observed point
        """
        distance = self.distance_difference_one_point(x, y)
        self.weight.fill(1.0)
        self.weight *= distance
        self.weight += 1.e-300
        self.weight /= sum(self.weight)

    def estimate(self):
        """
        Weighted average of the particles
        :return: estimated point
        """
        x_mean = np.average(self.particles_event[:, 1], weights=self.weight, axis=0).astype(int)
        y_mean = np.average(self.particles_event[:, 0], weights=self.weight, axis=0).astype(int)
        return x_mean, y_mean

    def new_resample(self):
        """
        Sampling Importance Resampling
        """
        indexes = systematic_resample(weights=self.weight)
        self.particles_event[:] = self.particles_event[indexes]
        self.weight.resize(len(self.particles_event))
        self.weight.fill(1.0 / len(self.weight))

    def return_effective_N(self):
        """
        Calculates the effective particles
        :return: effective particles
        """
        return 1.0 / np.sum(np.square(self.weight))

    def draw_particles(self, frame, x, y, color, color2):
        """
        Draw current particles, initial particles, trajectory of particles and link to measurements
        :param frame: Image frame
        :param x: x coordinate observed point
        :param y: y coordinate observed point
        :param color2: Color array
        :param color: color array
        :return:
        """
        for count in range(5):
            x_particle = math.floor(self.particles_event[count][0])
            y_particle = math.floor(self.particles_event[count][1])
            x_prev = math.floor(self.initial[count][0])
            y_prev = math.floor(self.initial[count][1])
            cv2.arrowedLine(frame, (x_particle, y_particle), (math.floor(x), math.floor(y)), [255, 255, 0], 1, 1, 0, 0.1)
            cv2.arrowedLine(frame, (x_particle, y_particle), (x_prev, y_prev), color2, 1, 8, 0, 0.1)
            cv2.circle(frame, (x_particle, y_particle), 1, color, 1)
