import cv2
import numpy as np

# Create a window to display the image
cv2.namedWindow('Real-time Image', cv2.WINDOW_NORMAL)

# Loop for displaying the image in real-time
while True:
    try:
        image = np.load("hand.npy")
        print(image.shape)
    except:
        pass
    # Display the image
    cv2.imshow('Real-time Image', image)

    # Check for key press and break the loop if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
