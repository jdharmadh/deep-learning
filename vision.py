import cv2
import numpy as np

# Load the image
image_path = "./big1.png"
image = cv2.imread(image_path)

# Resize to ensure consistent cell size (if needed)
grid_size = 5
resize_dim = 800
image_resized = cv2.resize(image, (resize_dim, resize_dim))
# Convert to HSV for better color segmentation
hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

# Define HSV color masks
color_masks = {
    1: [  # Red (two ranges to handle hue wrap-around)
        ((0, 70, 50), (10, 255, 255)),
        ((170, 70, 50), (180, 255, 255))
    ],
    2: [((100, 150, 50), (130, 255, 255))],  # Blue
    3: [((0, 0, 0), (180, 255, 50))],        # Black
    4: [((40, 70, 50), (80, 255, 255))]      # Green
}

# Initialize grid
grid = np.zeros((grid_size, grid_size), dtype=int)
# Calculate cell dimensions
cell_h = image_resized.shape[0] // grid_size
cell_w = image_resized.shape[1] // grid_size

# Process each cell
for i in range(grid_size):
    for j in range(grid_size):
        cell = hsv[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
        color_counts = {}

        for label, ranges in color_masks.items():
            mask_total = np.zeros(cell.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.inRange(cell, np.array(lower), np.array(upper))
                mask_total = cv2.bitwise_or(mask_total, mask)
            color_counts[label] = np.sum(mask_total)

        # Assign the color with the most detected pixels
        grid[i, j] = max(color_counts, key=color_counts.get)

# Print the detected grid
print(grid)

# Create a visualization of the detected grid
vis_grid = np.zeros((cell_h * grid_size, cell_w * grid_size, 3), dtype=np.uint8)

# Color mapping (value to BGR)
color_map = {
    1: (0, 0, 255),    # Red
    2: (255, 0, 0),    # Blue
    3: (0, 0, 0),      # Black
    4: (0, 255, 0)     # Green
}

# Fill in the visualization grid
for i in range(grid_size):
    for j in range(grid_size):
        color = color_map[grid[i, j]]
        vis_grid[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w] = color

# Create a side-by-side comparison
comparison = np.hstack((image_resized, vis_grid))

# Display the result
cv2.imshow("Original vs Detected Grid", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()
