🐦 Bird Sound Classification using Deep Learning
📌 Project Overview
This project builds a Bird Sound Classification System using Deep Learning (CNN). The system analyzes uploaded audio recordings of bird calls and predicts the bird species.

The model extracts Mel Spectrogram features from audio signals and feeds them into a Convolutional Neural Network (CNN) for classification.

The project also includes a Streamlit web interface where users can upload audio files (.mp3 / .wav) and instantly get predictions.

🎯 Objectives
Detect bird species from audio recordings.
Extract audio features using Librosa.
Train a CNN-based classifier using TensorFlow/Keras.
Provide an interactive Streamlit frontend for real-time prediction.
📂 Project Structure
Audio_Classification_Fixed
│
├── app.py                     # Streamlit frontend
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
│
├── data
│   └── Voice of Birds
│        ├── Crow
│        ├── Sparrow
│        └── Parrot
│
└── model
    ├── preprocessing.py       # Audio feature extraction
    ├── Data_preparation.py    # Dataset preparation & splitting
    ├── train_model.py         # CNN training script
    ├── best_model.keras       # Saved trained model
    ├── features_mel.npy
    ├── labels_mel.npy
    └── label_encoder.npy
🧠 Technologies Used
Python
TensorFlow / Keras
Librosa
NumPy
Scikit-learn
Matplotlib
Streamlit
📊 Dataset
The dataset consists of bird audio recordings collected from:

Xeno-Canto Bird Sound Database

Example query used:

https://xeno-canto.org/explore?query=Corvus%20enca

Bird classes used in this project:

Crow
Sparrow
Parrot
Audio recordings were downloaded in MP3 format and organized into class folders.

Example dataset structure:

data/Voice of Birds/

Crow/
crow1.mp3
crow2.mp3

Sparrow/
sparrow1.mp3
sparrow2.mp3

Parrot/
parrot1.mp3
parrot2.mp3
⚙️ Installation
1️⃣ Clone the repository
git clone <your-repository-url>
cd Audio_Classification_Fixed
2️⃣ Create virtual environment
python -m venv .venv
Activate environment:

.\.venv\Scripts\activate
3️⃣ Install dependencies
pip install -r requirements.txt
🚀 Running the Project
Step 1 – Extract audio features
python model/preprocessing.py
Step 2 – Prepare dataset
python model/Data_preparation.py
Step 3 – Train the CNN model
python model/train_model.py
Step 4 – Run the web interface
streamlit run app.py
Open in browser:

http://localhost:8501
🖥️ Web Interface
The Streamlit interface allows users to:

Upload .mp3 or .wav audio files
Process the audio
Predict bird species
Example output:

Predicted Bird: Crow
Confidence: 87%
🔍 Feature Extraction
Audio recordings are converted into Mel Spectrograms using Librosa:

Sampling Rate: 22050 Hz
Mel Bands: 128
Time Frames: 130
These spectrograms serve as input to the CNN model.

🧪 Model Architecture
The model uses a Convolutional Neural Network with:

Convolution Layers
MaxPooling Layers
Dense Layers
Softmax output for classification
📈 Future Improvements
Increase dataset size
Add more bird species
Improve CNN architecture
Deploy model online
Add spectrogram visualization in UI
👩‍💻 Author
Priyanka Sharma B.Tech – Computer Science and Engineering(Artificial Intelligence)

📜 License
This project is for educational and research purposes only.
