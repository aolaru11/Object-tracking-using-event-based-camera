# Object-tracking-using-event-based-camera

### The project is created for the Research Project course, taken at Delft University of Technology.


This model uses two clustering methods: Density-Based Spatial Clustering of Applications with Nois(DBSCAN) and Mean Shift algorithm.

The documentation of DBSCAN can be accessed using the link: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

The documentation of DBSCAN can be accessed using the link: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html

Also, Particle filter is implemented to correct the trajectory and its implementation can be seen in the ParticleFilter class.

The model is created to analyse the event-based data and the events were hadled using the class eventvision from the repository https://github.com/gorchard/event-Python.
