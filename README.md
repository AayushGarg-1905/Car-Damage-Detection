##  Car Damage Detection App [https://car-damage-detection-aayush-1905.streamlit.app/]

This app allows you to **drag and drop an image of a car**, and it will classify the type of damage present.

**Note:** The model is trained on **third-quarter front and rear views**. Please ensure the uploaded image captures the car from this angle for accurate predictions.

---

### Model Details & Development

* Dataset: \~2,300 labeled images with 6 target classes:

  1. Front Normal
  2. Front Crushed
  3. Front Breakage
  4. Rear Normal
  5. Rear Crushed
  6. Rear Breakage

* Model Development Journey:

  1. **Custom CNN**: Started with a basic Convolutional Neural Network to set a performance baseline.
  2. **Regularization**: Added Dropout and Data Augmentation to improve generalization.
  3. **EfficientNet**: Tried using EfficientNet for better accuracy but faced issues with generalization due to dataset limitations.
  4. **ResNet50 (Final)**: Used transfer learning with ResNet50 and fine-tuned the final layers. This yielded the best validation accuracy of around **80%**.

---

###  Setup Instructions

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

---

## YOLOv5 Integration in Car Damage Detection System

To enhance the functionality of the car damage detection system,YOLOv5 is integrated to validate whether the uploaded image actually contains a car. This pre-validation step ensures that only relevant images are processed further for damage classification.

### Why YOLOv5?

* **Car Presence Detection**: YOLOv5 is used to detect whether a car is present in the uploaded image. This is important to filter out irrelevant or incorrect uploads.
* **Robust Error Handling**: If YOLOv5 makes a wrong prediction (e.g., doesn't detect a car in a valid image or detects incorrectly), the system flags the case for manual review or prompts the user to upload the image again.
* **Multiple Cars Handling**: In scenarios where multiple cars are detected, the one occupying the largest portion of the image is selected for further processing. This ensures that the most prominent car is considered for damage detection.<br/>
Code for YOLO integration is present here:https://github.com/AayushGarg-1905/Car-Damage-Detection/blob/feature/yolo-integration
