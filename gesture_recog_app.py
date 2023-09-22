import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Define the class labels
class_labels = ["01_palm", "02_I", "03_fist", "04_fist_moving", "05_thumb", "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"]  # Replace with your actual class labels

# Load the trained model
model = tf.keras.models.load_model("model_v1.h5")  # Replace with your model file path

# Function to preprocess and classify an uploaded image
def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            image = Image.open(file_path)
            image = image.resize((64, 64))  # Resize the image to (64, 64)
            image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
            
            # Ensure the image has 3 color channels (RGB)
            if len(image.shape) == 2:
                image = np.stack((image,) * 3, axis=-1)
            
            image = image.reshape(1, 64, 64, 3)  # Reshape for model input
            prediction = model.predict(image)
            predicted_class = class_labels[np.argmax(prediction)]
            result_label.config(text=f"Predicted Class: {predicted_class}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create the main application window
app = tk.Tk()
app.title("Image Classification App")

# Create a button to upload an image
upload_button = tk.Button(app, text="Upload Image", command=classify_image)
upload_button.pack(pady=20)

# Create a label to display the result
result_label = tk.Label(app, text="", font=("Helvetica", 16))
result_label.pack()

# Start the GUI main loop
app.mainloop()