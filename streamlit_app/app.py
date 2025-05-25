# import streamlit as st
# from model_helper import predict

# st.title("Car Damage Detection")

# uploaded_file = st.file_uploader("Upload the file",type=["jpg","png","jpeg"])

# if uploaded_file:
#     image_path = "temp_file.jpg"
#     with open(image_path,"wb") as f:
#         f.write(uploaded_file.getbuffer())
#         st.image(uploaded_file, caption="Uploaded File",use_container_width=True)
#         prediction = predict(image_path)
#         st.info(f"Predicted Class: {prediction}")



import streamlit as st
from model_helper import predict
import torch
from PIL import Image

st.title("Car Damage Detection")

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_path = "temp_file.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file, caption="Uploaded File", use_container_width=True)


    prediction_before_yolo = predict(image_path)


    model =torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
    results = model(image_path)
    st.image(results.render()[0], caption="YOLOv5 Detection Result", use_container_width=True)


    class_names = results.names
    car_class_index = truck_class_index = motorcycle_class_index = None
    for i, name in class_names.items():
        if name == "car":
            car_class_index = i
        elif name == "truck":
            truck_class_index = i
        elif name == "motorcycle":
            motorcycle_class_index = i

    if car_class_index is None:
        st.error("YOLOv5 model does not support 'car' class.")
    else:
        detected_labels = results.pred[0][:, -1].cpu().numpy()
        detected_boxes = results.pred[0][:, :4].cpu().numpy()
        confidences = results.pred[0][:, 4].cpu().numpy()


        other_threshold = 0.6


        valid_indices = []
        for i, (label, conf) in enumerate(zip(detected_labels, confidences)):
            if label == car_class_index:
                valid_indices.append(i)
            elif label in [truck_class_index, motorcycle_class_index] and conf > other_threshold:
                pass

        if len(valid_indices) == 0:
            st.warning("No valid car detection found. Please upload a clearer image.")
        else:

            st.info(f"Predicted Damage Class before YOLO filtering: {prediction_before_yolo}")
            max_area = -1
            max_index = -1
            for i in valid_indices:
                if detected_labels[i] != car_class_index:
                    continue  
                x1, y1, x2, y2 = detected_boxes[i]
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    max_index = i

            if max_index == -1:
                st.error("No 'car' detected among valid detections.")
            else:
                st.write(f"Selected car detection with bounding box area: {max_area:.0f} pixels and confidence: {confidences[max_index]:.2f}")

                # Cropping image
                img = Image.open(image_path).convert("RGB")
                x1, y1, x2, y2 = detected_boxes[max_index].astype(int)
                car_crop = img.crop((x1, y1, x2, y2))
                car_crop.save("cropped_car.jpg")
                yolo_detected_label = class_names[int(detected_labels[max_index])]

                prediction_after_yolo = predict("cropped_car.jpg")
                st.info(f"Predicted Damage Class: {prediction_after_yolo}")
                st.image("cropped_car.jpg", caption="Image used for detection", use_container_width=True)
                if prediction_before_yolo != prediction_after_yolo or yolo_detected_label != "car":
                    st.warning("⚠️ We noticed a mismatch between predictions or the detected object is not a 'car'.")

                    st.image("cropped_car.jpg", caption="Detected object used for prediction")

                    user_feedback = st.radio("Is this image showing your actual damaged car?", ["Yes", "No"])

                    if user_feedback == "No":
                        st.info("Please upload another image where the damaged region is clearly visible — ideally taken from the third rear or front quarter view and ensure only one vehicle is present in image")
                        st.stop()  
                    else:
                        st.success("Thanks for confirming. Your claim will be fast-tracked for processing.")
                if len(valid_indices) > 1:
                    st.warning(f"Multiple detections found, using largest car for analysis.")

