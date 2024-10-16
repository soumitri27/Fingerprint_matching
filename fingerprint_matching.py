import os
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter import filedialog


def choose_folder():
    root = Tk()
    root.withdraw()  
    folder_path = filedialog.askdirectory(title="Select the folder containing fingerprint images")
    return folder_path


def choose_file():
    root = Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(title="Select the sample fingerprint image", 
                                           filetypes=[("Image Files", "*.bmp;*.jpg;*.png")])
    return file_path


print("Select the sample fingerprint image:")
sample_image_path = choose_file()


print("Select the folder containing the real fingerprint images:")
real_folder = choose_folder()


sample = cv2.imread(sample_image_path)


if sample is None:
    raise FileNotFoundError("Sample image not found or could not be loaded.")

best_score = 0
filename = None
image = None
kp1, kp2, mp = None, None, None


for file in os.listdir(real_folder):
    fingerprint_image_path = os.path.join(real_folder, file)
    fingerprint_image = cv2.imread(fingerprint_image_path)
    
    if fingerprint_image is None:
        print(f"Image {file} could not be loaded.")
        continue
    
    
    sift = cv2.SIFT_create()

    
    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)
    
    if descriptors_1 is None or descriptors_2 is None:
        print(f"No descriptors found for {file}.")
        continue


    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict(checks=50))
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    match_points = []

 
    for p, q in matches:
        if p.distance < 0.75 * q.distance:
            match_points.append(p)

    keypoint = min(len(keypoints_1), len(keypoints_2))

    if keypoint == 0: 
        continue

    score = len(match_points) / keypoint * 100

    if score > best_score:
        best_score = score
        filename = file
        image = fingerprint_image
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points


if image is not None:
    result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
    
    print("BEST MATCH: " + filename)
    print("SCORE: " + str(best_score))
    
    
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

   
    plt.imshow(result_rgb)
    plt.title("Matching Result")
    plt.axis('off')  
    plt.show()
else:
    print("No good match found.")
