import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import numpy as np
from training_logic import Train


class DigitPredictorApp:
    def __init__(self, root, training_set):
        self.root = root
        self.training_set = training_set
        self.root.title("Digit Predictor")

        self.canvas_width = 200
        self.canvas_height = 200
        self.bg_color = "white"
        self.paint_color = "black"
        self.radius = 5

        self.init_gui()

    def init_gui(self):
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg=self.bg_color)
        self.canvas.pack(side=tk.RIGHT)

        self.predict_button = ttk.Button(self.root, text="Predict", command=self.predict)
        self.predict_button.pack(side=tk.BOTTOM)

        self.prediction_var = tk.StringVar()
        self.prediction_label = ttk.Label(self.root, textvariable=self.prediction_var, font=("Helvetica", 48))
        self.prediction_label.pack(side=tk.LEFT)

        self.erase_button = ttk.Button(self.root, text="Erase", command=self.clear_canvas)
        self.erase_button.pack(side=tk.BOTTOM)


        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

    def clear_canvas(self):
        self.canvas.delete("all")

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.prediction_var.set("")


    def paint(self, event):
        x1, y1 = (event.x - self.radius), (event.y - self.radius)
        x2, y2 = (event.x + self.radius), (event.y + self.radius)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.paint_color, outline=self.paint_color)
        self.draw.ellipse([x1, y1, x2, y2], fill=self.paint_color)

    def predict(self):
        self.image_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(self.image_resized, dtype=np.float32).reshape(1, 28*28) / 255.0
        img_array = 1.0 - img_array 
        prediction = training_set.prediction(img_array)
        prediction_value = prediction[0][0]
        self.prediction_var.set(str(prediction_value))


if __name__ == "__main__":
    training_set = Train(1000)
    root = tk.Tk()
    app = DigitPredictorApp(root, training_set)
    root.mainloop()

    