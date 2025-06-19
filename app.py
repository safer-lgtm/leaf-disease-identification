from taipy.gui import Gui
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Falls du die class_indices vom train_generator gespeichert hast
class_names = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry___Powdery_mildew',
    6: 'Cherry___healthy',
    7: 'Corn___Cercospora_leaf_spot',
    8: 'Corn___Common_rust',
    9: 'Corn___Northern_Leaf_Blight',
    10: 'Corn___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

# Modell laden
model = load_model("best_model.keras")

def predict_image(model, path_to_img):
    img = load_img(path_to_img, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    probs = model.predict(img)
    top_prob = probs.max()
    top_pred = class_names[np.argmax(probs)]

    return top_prob, top_pred

# GUI-Logik
content = ""
img_path = "placeholder_image.png"
prob = 0
pred = ""

index = """
<|text-center|
<|{"logo.png"}|image|width=25vw|>

<|{content}|file_selector|extensions=.jpg,.jpeg,.png|label=Bild auswählen|>

<|{pred}|>

<|{img_path}|image|width=25vw|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>
"""

def on_change(state, var_name, var_val):
    if var_name == "content":
        top_prob, top_pred = predict_image(model, var_val)
        state.prob = round(top_prob * 100)
        state.pred = f"Dies ist vermutlich: **{top_pred}**"
        state.img_path = var_val

app = Gui(page=index)

if __name__ == "__main__":
    app.run(use_reloader=True)