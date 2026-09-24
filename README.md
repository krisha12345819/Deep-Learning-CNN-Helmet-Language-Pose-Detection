# 🤖 Teachable Machine AI Projects

A collection of **3 machine learning mini-projects** built using [Google Teachable Machine](https://teachablemachine.withgoogle.com/) and run locally with **TensorFlow / Keras** in Python (Jupyter Notebooks) — no browser required! 🚀

Each project is trained on **Teachable Machine**, exported, and then loaded into a standalone Python notebook for real-world inference (images, audio, and pose).

---

## 📁 Repository Structure

```
📦 Teachable-Machine-AI-Projects
┣ 📂 1-Helmet-Detection
┃ ┣ 📜 helmet_detection_image_prediction.ipynb
┃ ┣ 🧠 keras_model.h5
┃ ┣ 🏷️ labels.txt
┃ ┗ 🖼️ test.jpeg
┃
┣ 📂 2-Language-Detection
┃ ┣ 📜 test_teachable_audio_model.ipynb
┃ ┣ 🧠 model.json
┃ ┣ ⚖️ weights.bin
┃ ┣ 🗂️ metadata.json
┃ ┗ 🎛️ languges.tm
┃
┗ 📂 3-Pose-Detection
  ┣ 📜 pose_model_inference.ipynb
  ┣ 🧠 model.json
  ┣ ⚖️ weights.bin
  ┣ 🗂️ metadata.json
  ┣ 🎛️ pose_model.tm
  ┗ 🖼️ test.jpeg
```

---

## 🪖 1️⃣ Helmet Detection

> 🎯 Detects whether a person in an image is **wearing a helmet**, **not wearing a helmet**, or if **no person** is present at all — a great use case for road-safety & workplace-safety monitoring.

### 🧠 How it works
- Model trained on **Google Teachable Machine (Image Project)** and exported as a Keras `.h5` model.
- Loaded locally using `tf_keras.models.load_model()`.
- Input image is resized to **224×224**, normalized to `[-1, 1]`, and passed through the model.
- Outputs a class prediction + confidence score.

### 🏷️ Classes
| Label ID | Class Name |
|:--------:|:-----------|
| 0 | 🪖 Helmet |
| 1 | 🚫 No Person |
| 2 | ⚠️ Person, No Helmet |

### 📦 Files
- `helmet_detection_image_prediction.ipynb` – Main inference notebook
- `keras_model.h5` – Trained Teachable Machine model
- `labels.txt` – Class label mapping

### ▶️ Run it
```bash
pip install tf-keras pillow opencv-python numpy
```
1. Place `keras_model.h5`, `labels.txt`, and your test image in the same folder.
2. Open `helmet_detection_image_prediction.ipynb`.
3. Update `IMAGE_PATH` with your image.
4. Run all cells — get the predicted class + confidence % 🎉

### 📊 Sample Output
```
Predicted Class: Person_no_helmet
Confidence: 81.44%
```

---

## 🗣️ 2️⃣ Language Detection (Audio)

> 🎯 Classifies **spoken audio** into **English 🇬🇧**, **Gujarati 🙏**, **Hindi 🇮🇳**, or **Background Noise 🔇** using a Teachable Machine **Audio Project**.

### 🧠 How it works
- Trained with **Teachable Machine's Speech Commands** model (TFJS v0.4.0).
- Audio is converted into a spectrogram using `librosa`, matching the exact preprocessing Teachable Machine uses in the browser.
- The exported `model.json` + `weights.bin` (TensorFlow.js format) are rebuilt as a Keras model in Python and run on the spectrogram.
- Supports both **prerecorded audio files** and **live microphone input** 🎙️.

### 🏷️ Classes
| Label | Meaning |
|:------|:--------|
| 🔇 Background Noise | Silence / ambient noise |
| 🇬🇧 english | English speech |
| 🙏 gujarati | Gujarati speech |
| 🇮🇳 hindi | Hindi speech |

### 📦 Files
- `test_teachable_audio_model.ipynb` – Main inference notebook
- `model.json` – Model architecture (TFJS Speech Commands, `TMv2`)
- `weights.bin` – Trained weights
- `metadata.json` – Word labels & TM version info
- `languges.tm` – Original Teachable Machine project file (not needed for inference)

### ▶️ Run it
```bash
pip install tensorflow librosa numpy matplotlib sounddevice soundfile
```
1. Place `model.json`, `weights.bin`, and `metadata.json` in the same folder as the notebook.
2. Open `test_teachable_audio_model.ipynb` and run all cells.
3. Test with an audio file, **or** record live from your microphone 🎤.

---

## 🕺 3️⃣ Pose Detection

> 🎯 Classifies **human body poses** from an image into custom gesture classes using a Teachable Machine **Pose Project**.

### 🧠 How it works
1. **PoseNet** (MobileNetV1, multiplier `0.75`, stride `16`, input `257×257`) extracts pose heatmaps (17 channels) + offsets (34 channels) on a 17×17 grid.
2. These are flattened into a single **14,739-value** feature vector.
3. The custom Teachable Machine head — `Dense(100, relu) → Dense(4, softmax)` — classifies the pose vector.
4. Since the exported model only contains the classifier head, the notebook downloads the **PoseNet backbone** separately and rebuilds the full pipeline in Python.

### 🏷️ Classes
| Label | Gesture |
|:------|:--------|
| ✅ yes | "Yes" pose |
| ❌ No | "No" pose |
| ✋ Stop | "Stop" pose |
| 🆘 Help | "Help" pose |

### 📦 Files
- `pose_model_inference.ipynb` – Main inference notebook
- `model.json` – Classifier head architecture
- `weights.bin` – Trained weights
- `metadata.json` – Labels & PoseNet settings
- `pose_model.tm` – Original Teachable Machine project file
- `test.jpeg` – Sample test image

### ▶️ Run it
```bash
pip install tensorflow pillow matplotlib numpy
```
1. Place `model.json`, `weights.bin`, `metadata.json`, and `test.jpeg` in the same folder as the notebook.
2. Open `pose_model_inference.ipynb` and run all cells.
3. ✅ Optional: verify results against the browser using `pose_predictor.html` + a local server (`python -m http.server 8000`).

> ⚠️ **Note:** Teachable Machine models are trained on a small number of samples, so confidence on new images/backgrounds may occasionally be low or off — this is expected behavior, not a bug.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-tf.keras-red?logo=keras&logoColor=white)
![TeachableMachine](https://img.shields.io/badge/Google-Teachable%20Machine-yellow?logo=google&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)

| Tool | Used For |
|------|----------|
| 🧠 Google Teachable Machine | Training all 3 models (Image / Audio / Pose) |
| 🔥 TensorFlow / tf-keras | Running inference locally in Python |
| 🎧 Librosa | Audio preprocessing for language detection |
| 🖼️ Pillow / OpenCV | Image preprocessing for helmet & pose detection |
| 📓 Jupyter Notebook | Interactive experimentation & visualization |

---

## 🚀 Quick Start (All Projects)

```bash
# Clone the repo
git clone <your-repo-url>
cd Teachable-Machine-AI-Projects

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install all dependencies
pip install tensorflow tf-keras pillow opencv-python numpy librosa matplotlib sounddevice soundfile

# Launch Jupyter
jupyter notebook
```
Then open the notebook inside the project folder you want to run ✅

---

## 🙋‍♀️ Author

Made with ❤️ and lots of ☕ while learning Machine Learning & Computer Vision.

⭐ If you found this useful, consider giving the repo a star!

---

## 📄 License

This project is open-source and available for learning purposes. Feel free to fork, modify, and build upon it! 🎓
