import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('/your/file/path/asl-cnn-v2.keras')  # Ensure this path is correct

def preprocess_image_for_prediction(img, target_size=(28, 28)):
    """
    Preprocess the input image for prediction.
    """
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def predict_sign_language(img):
    """
    Predict the sign language gesture from the processed image.
    """
    processed_img = preprocess_image_for_prediction(img)
    predictions = model.predict(processed_img)
    pred_class = np.argmax(predictions, axis=1)[0]
    pred_certainty = np.max(predictions)
    return pred_class, pred_certainty

def main():
    cam_capture = cv2.VideoCapture(0)
    if not cam_capture.isOpened():
        print("Error opening video stream or file")
        return

    x_start, y_start = 100, 100
    roi_width, roi_height = 400, 400

    while True:
        ret, frame = cam_capture.read()
        if not ret:
            break

        cv2.rectangle(frame, (x_start, y_start), (x_start + roi_width, y_start + roi_height), (0, 255, 0), 2)
        roi = frame[y_start:y_start + roi_height, x_start:x_start + roi_width]

        pred_class, pred_certainty = predict_sign_language(roi)
        label = chr(pred_class + 65)

        cv2.putText(frame, f'Pred: {label}, Cert: {pred_certainty:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Sign Language Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
