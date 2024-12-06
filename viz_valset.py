import pickle
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


class ColorStyle:
    def __init__(self):
        self.color = [(252, 176, 243), (252, 176, 243), (252, 176, 243),
                      (0, 176, 240), (0, 176, 240), (0, 176, 240),
                      (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127),
                      (255, 255, 0), (255, 255, 0), (169, 209, 142),
                      (169, 209, 142), (169, 209, 142)]

        self.link_pairs = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                           [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                           [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]]

        self.point_color = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
                            (240, 2, 127), (240, 2, 127),
                            (255, 255, 0), (169, 209, 142),
                            (255, 255, 0), (169, 209, 142),
                            (255, 255, 0), (169, 209, 142),
                            (252, 176, 243), (0, 176, 240), (252, 176, 243),
                            (0, 176, 240), (252, 176, 243), (0, 176, 240)]

        # Normalize colors to [0,1] range
        self.link_pairs_with_color = []
        for i in range(len(self.color)):
            self.link_pairs_with_color.append(tuple(np.array(self.color[i]) / 255.))

        self.ring_color = []
        for color in self.point_color:
            self.ring_color.append(tuple(np.array(color) / 255.))

        # Define color IDs
        color_ids = [(0, 176, 240), (252, 176, 243), (169, 209, 142),
                     (255, 255, 0), (240, 2, 127)]
        self.color_ids = [tuple(np.array(c) / 255.) for c in color_ids]


def visualize_keypoints(input_img, keypoints, color_style=None):
    if color_style is None:
        color_style = ColorStyle()

    vis_img = input_img
    h, w = vis_img.shape[:2]

    # Create figure
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = plt.subplot(1, 1, 1)

    # Show background image
    bk = plt.imshow(vis_img)
    bk.set_zorder(-1)

    # Draw skeleton links
    for i, pair in enumerate(color_style.link_pairs):
        if keypoints[pair[0], 2] > 0 and keypoints[pair[1], 2] > 0:
            line = mlines.Line2D(
                np.array([keypoints[pair[0], 0], keypoints[pair[1], 0]]),
                np.array([keypoints[pair[0], 1], keypoints[pair[1], 1]]),
                ls='-', lw=2, alpha=1,
                color=color_style.link_pairs_with_color[i % len(color_style.link_pairs_with_color)]
            )
            line.set_zorder(0)
            ax.add_line(line)

    # Draw keypoints
    for j in range(keypoints.shape[0]):
        if keypoints[j, 2] > 0:  # confidence check
            circle = mpatches.Circle(
                (keypoints[j, 0], keypoints[j, 1]),
                radius=3,
                ec='black',
                fc=color_style.ring_color[j % len(color_style.ring_color)],
                alpha=1,
                linewidth=1
            )
            circle.set_zorder(1)
            ax.add_patch(circle)

            # Add keypoint number
            plt.text(keypoints[j, 0] + 5, keypoints[j, 1] + 5,
                     str(j),
                     color='white',
                     bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=1),
                     fontsize=8,
                     ha='left',
                     va='bottom')

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    return fig


# val_label.pkl file path
val_label_path = 'D:/Dev/Dataset/coco/annotations/val_label.pkl'
base_path = 'D:/Dev/Dataset/coco/val2017'

# load file
with open(val_label_path, 'rb') as f:
    val_label_data = pickle.load(f)

# Create a window for displaying images
plt.ion()  # Interactive mode on

for i in range(len(val_label_data)):
    keypoints = val_label_data[i]['keypoints']
    img_path = os.path.join(base_path, val_label_data[i]['img_file'])
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Visualize keypoints
    fig = visualize_keypoints(img, keypoints)
    plt.show()

    # Wait for key press
    key = input("Press Enter to continue (or 'q' to quit): ")
    if key.lower() == 'q':
        break

    plt.close(fig)

plt.ioff()  # Interactive mode off