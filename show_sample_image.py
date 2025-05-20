import urllib.request
import cv2
import matplotlib.pyplot as plt
import os

# New sample image URL (Wikimedia Commons, public domain)
url = "https://assets.nhs.uk/nhsuk-cms/images/A_1017__superficial-spreading-malignant-melan.width-1534.jpg"
filename = "sample.jpg"

# Download the image only if it doesn't already exist
if not os.path.exists(filename):
    urllib.request.urlretrieve(url, filename)

# Load the image with OpenCV
img = cv2.imread(filename)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(img_rgb)
plt.title("Sample Skin Lesion Image")
plt.axis("off")
plt.show()
