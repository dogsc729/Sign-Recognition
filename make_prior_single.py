import argparse
import cv2
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="prior map")
    parser.add_argument("--img", type=str, help="image file")
    args = parser.parse_args()
    
    # Set window size for prior calculation
    window_size = 3
    
    # Read input image
    img_path = args.img
    img = cv2.imread(img_path, 0)
    
    # Normalize input image
    img = (img / float(max(img.max(), 1))).astype(np.float32)
    
    # If input image is all zeros, create an empty prior image
    if img.sum() == 0:
        prior_img = np.zeros_like(img)
    else:
        prior_img = img
    
    # Pad image to handle edge cases
    prior_img = np.pad(prior_img, ((window_size//2, window_size//2), (window_size//2, window_size//2)), mode='constant')
    print(prior_img)
    output = np.zeros_like(prior_img)
    # Calculate prior image by taking the mean of neighboring pixels
    for i in range(prior_img.shape[0]):
        for j in range(prior_img.shape[1]):
            window = prior_img[i: i+3, j:j+3]
            mean_value = np.mean(window)
            output[i, j] = mean_value

    # Scale prior image to [0, 255] and convert to integer
    output = (255 * output).astype(np.uint8)
    print(output)
    # Save prior image
    prior_img_path = "./prior_test_output.jpg"
    cv2.imwrite(prior_img_path, output)
    
    print("Prior image saved:", prior_img_path)
    
    return 0

if __name__ == "__main__":
    main()
