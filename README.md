# EcoSort Prime | Smart Waste Analytics

An AI-powered waste segregation dashboard built to classify waste in real-time using computer vision. This application utilizes a live camera feed to scan waste items and automatically categorizes them into appropriate disposal bins (Wet, Dry, or Domestic Hazardous) to optimize recycling and waste management workflows.

Developed as an initiative for the Synapse AI Club at MIT World Peace University.

## Features
* **Real-Time Scanning:** Uses Streamlit's camera input to capture and analyze waste items instantly.
* **Deep Learning Engine:** Powered by a Convolutional Neural Network (CNN) utilizing MobileNetV2 transfer learning via TensorFlow/Keras.
* **Smart Categorization:** The model classifies 10 specific waste types (battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash) and maps them to 3 actionable categories:
  * **Wet Bin** (Compost / Biological)
  * **Dry Bin** (Recyclables / General)
  * **Hazardous Bin** (E-Waste / Batteries)
* **Custom UI:** A responsive, dark-themed "luxury dashboard" interface with dynamic styling based on the prediction.

## Technology Stack
* **Frontend:** Streamlit, Custom HTML/CSS
* **Backend Framework:** Python
* **Machine Learning:** TensorFlow 2.x, TensorFlow Hub, Keras
* **Image Processing:** Pillow (PIL), NumPy

## Project Structure
```text
├── app.py                  # Main Streamlit application file
├── waste_model_v1/         # TensorFlow SavedModel directory (Graph & Weights)
│   ├── saved_model.pb      
│   ├── variables/          
│   └── assets/
```
How to Run Locally
1. Clone the repository

Bash
git clone [https://github.com/yourusername/ecosort-prime.git](https://github.com/yourusername/ecosort-prime.git)
cd ecosort-prime
2. Create a virtual environment (Recommended)

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
3. Install dependencies

Bash
pip install -r requirements.txt
4. Run the application

Bash
streamlit run app.py
The application will automatically open in your default web browser at http://localhost:8501. Grant camera permissions when prompted to use the scanning feature.

Model Architecture details
The core model is built using Transfer Learning. We utilized the pre-trained MobileNetV2 architecture as the base feature extractor, freezing its initial layers. A custom classification head (Dense layers) was added and fine-tuned on our specific waste dataset. The model expects input images of shape (224, 224, 3) and uses the Adam optimizer with Sparse Categorical Crossentropy loss.

# Author
Praneel Chhabria        
