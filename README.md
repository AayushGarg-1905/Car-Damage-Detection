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

