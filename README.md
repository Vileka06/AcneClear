# AcneClear AI

A simple Streamlit mini project for photo enhancement using **GFPGAN**.

## Features

1. Upload a photo.
2. Enhance the photo using GFPGAN.
3. Compare the original and enhanced photos.
4. Download the enhanced photo.

## Run the project

Use Python 3.11 for the best compatibility with the model dependencies.

```bash
git clone https://github.com/Vileka06/AcneClear.git
cd AcneClear

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit address shown in your terminal (usually http://localhost:8501).

The GFPGAN v1.3 model weights are downloaded automatically the first time you enhance a photo and stored in the local `models/` folder. Model weights and uploaded photos are not committed to this repository.

## Project structure

```text
AcneClear/
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
└── .gitignore
```

## Note

GFPGAN is a general face-restoration model. It can improve facial details, but it is not an acne detector, acne-removal guarantee, medical diagnosis, or treatment.
