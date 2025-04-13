import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Train images path
    train_images_path = os.path.join(os.path.dirname(__file__), "data/quickdraw_subset_np/train_images.npy")
    # Train labels path
    train_labels_path = os.path.join(os.path.dirname(__file__), "data/quickdraw_subset_np/train_labels.npy")

    # Test images path
    test_images_path = os.path.join(os.path.dirname(__file__), "data/quickdraw_subset_np/test_images.npy")
    # Test labels path
    test_labels_path = os.path.join(os.path.dirname(__file__), "data/quickdraw_subset_np/test_labels.npy")

    # Load the dataset
    train_images = np.load(train_images_path)
    train_labels = np.load(train_labels_path)
    test_images = np.load(test_images_path)
    test_labels = np.load(test_labels_path)

    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Distinct labels: {np.unique(train_labels)}")

    # Class images
    class_images = {}
    for class_label in np.unique(train_labels):
        class_images[class_label] = train_images[train_labels == class_label]

    # Set number of samples per class
    samples_per_class = 5
    num_classes = len(class_images)

        # Create a figure
    fig, axes = plt.subplots(nrows=num_classes, ncols=samples_per_class, figsize=(samples_per_class * 2, num_classes * 2))

    # Plot each class's images
    for row_idx, (class_label, images) in enumerate(class_images.items()):
        random_indices = np.random.choice(images.shape[0], size=samples_per_class, replace=False)
        for col_idx, idx in enumerate(random_indices):
            ax = axes[row_idx, col_idx]
            # Convert to binary image
            binary_image = (images[idx] > 181).astype(np.uint8)
            ax.imshow(binary_image, cmap='gray')
            ax.axis('off')
            if col_idx == 0:
                ax.set_title(f"Class {class_label}")

    plt.tight_layout()
    plt.show()
    # plt.show(block=False)
