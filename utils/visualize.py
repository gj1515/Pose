import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from typing import Optional, Union
import numpy as np
import torch
import cv2


def visualize_training_data(
        images: Union[torch.Tensor, np.ndarray],
        heat_maps_gt: Optional[Union[torch.Tensor, np.ndarray]] = None,
        wait_key: bool = True
) -> None:
    """
    Visualize training images and their corresponding heatmaps if available.

    Args:
        images: Input images tensor/array of shape [B, C, H, W]
        heat_maps_gt: Optional ground truth heatmaps tensor/array of shape [B, num_keypoints, H, W]
        wait_key: Whether to wait for key press before closing windows
    """
    # Print input shape and value range
    if isinstance(images, torch.Tensor):
        print('Input shape:', images.shape)
        # Get the first image and convert it
        image = images[0].cpu()  # Shape: [3, H, W]
        image = image.permute(1, 2, 0).numpy()  # Shape: [H, W, 3]
    else:
        print('Input shape:', images.shape)
        image = images[0]

    # Normalize if needed (assuming input range is [0,1] or [-1,1])
    if image.min() < 0:  # if normalized to [-1,1], convert to [0,1]
        image = (image + 1) / 2


    # Denormalize using ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std[None, None, :] * image + mean[None, None, :]

    # Convert to uint8 format (0-255)
    image = (image * 255).clip(0, 255).astype(np.uint8)

    # Convert RGB to BGR for cv2
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Display image
    cv2.imshow('Input Image', image_bgr)

    # Visualize heatmaps if they exist
    if heat_maps_gt is not None:
        if isinstance(heat_maps_gt, torch.Tensor):
            heatmaps = heat_maps_gt[0].cpu().detach().numpy()
        else:
            heatmaps = heat_maps_gt[0]

        num_keypoints = heatmaps.shape[0]

        # Create figure for individual heatmaps
        plt.figure(figsize=(20, 4))
        for joint_idx in range(num_keypoints):
            plt.subplot(1, num_keypoints, joint_idx + 1)

            # Plot heatmap
            plt.imshow(heatmaps[joint_idx], cmap='jet')
            plt.colorbar()
            plt.title(f'Joint {joint_idx}')
            plt.xticks([])
            plt.yticks([])

        plt.suptitle('Joint Heatmaps')
        plt.tight_layout()
        plt.show()

        # Create figure for heatmap overlays
        plt.figure(figsize=(60, 16))
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        for joint_idx in range(num_keypoints):
            plt.subplot(1, num_keypoints, joint_idx + 1)

            # Resize heatmap to match image dimensions
            resized_heatmap = cv2.resize(
                heatmaps[joint_idx],
                (rgb_image.shape[1], rgb_image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

            # Show original image with heatmap overlay
            plt.imshow(rgb_image)
            plt.imshow(resized_heatmap, cmap='jet', alpha=0.5)
            plt.colorbar()
            plt.title(f'Joint {joint_idx} Overlay')
            plt.xticks([])
            plt.yticks([])

        plt.suptitle('Joint Heatmap Overlays')
        plt.tight_layout()
        plt.show()

    if wait_key:
        key = cv2.waitKey(0)
        if key == 27:  # ESC key
            cv2.destroyAllWindows()



# val heatmap visualize
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

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    return fig

def visualize_heatmaps(image_ori, input_img, heatmap_out):
    """
    Visualize original image, input tensor image, and heatmaps for each keypoint

    Args:
        image_ori: Original image in BGR format
        input_img: Preprocessed input tensor image
        heatmap_out: Heatmap output from model (n_keypoints, height, width)
    """

    n_keypoints = heatmap_out.shape[0]
    rows = int(np.ceil(np.sqrt(n_keypoints + 2)))
    cols = int(np.ceil((n_keypoints + 2) / rows))

    plt.figure(figsize=(20, 20))

    # Plot original image
    plt.subplot(rows, cols, 1)
    plt.imshow(cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Plot input tensor image
    plt.subplot(rows, cols, 2)
    plt.imshow(input_img)
    plt.title('Input Tensor Image')
    plt.axis('off')

    # Plot heatmaps for each keypoint
    for i in range(n_keypoints):
        ax = plt.subplot(rows, cols, i + 3)

        # Resize heatmap to match input image size
        resized_heatmap = cv2.resize(heatmap_out[i],
                                     (input_img.shape[1], input_img.shape[0]),
                                     interpolation=cv2.INTER_CUBIC)

        # Overlay heatmap on input image
        plt.imshow(input_img)
        plt.imshow(resized_heatmap, cmap='jet', alpha=0.6)

        # Plot maximum point
        max_idx = np.unravel_index(np.argmax(resized_heatmap), resized_heatmap.shape)
        y, x = max_idx
        plt.plot(x, y, 'ko', markersize=5)
        plt.title(f'Keypoint {i}\nmax at ({x}, {y})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    return plt.gcf()
