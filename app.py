import cv2
import numpy as np
import streamlit as st

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

def detect_objects(img):
    height, width, _ = img.shape
    #scaling, resizing, mean value
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
#Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in np.array(indexes).flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = (255, 255, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    return img

def main():
    st.title("Object Detection Web App")
    st.write("Upload an image, video, or use the webcam for object detection.")

    option = st.sidebar.selectbox("Select Option", ["Image", "Video", "Webcam"])

    if option == "Image":
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, 1)
            st.image(image, channels="BGR", caption="Uploaded Image")
            st.write("---")
            st.write("Detecting objects...")
            result_image = detect_objects(image)
            st.image(result_image, channels="BGR", caption="Objects Detected Image")

    elif option == "Video":
        uploaded_video = st.file_uploader("Upload Video", type=["mp4"])
        if uploaded_video is not None:
            video = np.array(bytearray(uploaded_video.read()), dtype=np.uint8)
            video = cv2.imdecode(video, 1)
            st.video(uploaded_video, format="video/mp4")
            st.write("---")
            st.write("Detecting objects...")
            
            # Save video to a temporary file
            with st.spinner("Processing..."):
                temp_file = st.empty()
                temp_filename = "temp.mp4"
            #Opens file in write binary
                with open(temp_filename, "wb") as file:
                    file.write(uploaded_video.read())
                
                # Perform object detection on the video It allows reading frames from videos
                cap = cv2.VideoCapture(temp_filename)
                #VideoWriter to save processed video frames avc1 is video codec
                video_result = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'avc1'), 30, (int(cap.get(3)), int(cap.get(4))))
               #as long as frames can be read from video
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                #performs object detection on current frame
                    result_frame = detect_objects(frame)
                    video_result.write(result_frame)
                #releases video capture object
                cap.release()
                video_result.release()
            
            temp_file.info("Video processing completed!")
            st.video("output.mp4", format="video/mp4")

    elif option == "Webcam":
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
        #reads next frame from webcam
            _, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Perform object detection on the webcam frame
            detected_img = detect_objects(img)

            # Display the webcam frame with detected objects
            stframe.image(detected_img, caption="Webcam", channels="RGB", use_column_width=True)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
#runs app.py directly through script 
if __name__ == '__main__':
    main()
