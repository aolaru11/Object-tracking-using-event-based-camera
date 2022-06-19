import matplotlib.pyplot as plt
import cv2


def plot_multiple_frames(list_images):
    fig = plt.figure()
    rows = 2
    columns = len(list_images) // 2
    for i in range(1, len(list_images)):
        fig.add_subplot(rows, columns, i)
        plt.imshow(list_images[i])
        plt.axis('off')

    plt.show()


def show_resized_image(frame):
    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.imshow('original', frame)
    cv2.resizeWindow('original', 200, 200)
    cv2.waitKey(0)


def draw_bounding_box_tracking(x_estimated, y_estimated, frame, bounding_box_width, bounding_box_height):
    x_up = x_estimated - bounding_box_width
    y_up = y_estimated - bounding_box_height
    x_down = x_estimated + bounding_box_width
    y_down = y_estimated + bounding_box_height

    cv2.rectangle(frame, (x_up, y_up), (x_down, y_down), [255, 0, 0], 1)
    return frame