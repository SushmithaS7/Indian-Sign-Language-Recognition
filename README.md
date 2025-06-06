#  Indian Sign Language Recognition System

This project presents a **real-time Indian Sign Language (ISL) recognition system** designed to identify static gestures for **alphabets (A–Z)**, **digits (1–9)**, and **frequent phrases**, with integrated **text-to-speech (TTS)** support to enable auditory feedback.

The system aims to bridge the communication gap between the hearing-impaired community and the general public through an accessible, scalable, and interactive AI-powered solution.

---

##  Abstract

This project gives an overview of a real-time Indian Sign Language (ISL) recognition system that can identify alphabets (A–Z), numbers (1–9), and frequent phrases and has text-to-speech conversion incorporated to ease communication. In order to identify static hand gestures for alphabets and digits, the system uses a **Feedforward Neural Network (FNN)** that has been trained on a dataset of ISL gestures using **MediaPipe's hand tracking** solution with landmark data achieving **98.8% accuracy**. For dynamic phrase recognition, a **Long Short-Term Memory (LSTM)** model is employed, using pose sequences obtained from **MediaPipe's holistic pose estimation** pipeline to identify temporal patterns in sign videos with **accuracy of 74.19%**. Textual output, which includes recognized gestures, is then spoken via a **text-to-speech engine** with immediate auditory feedback. The system's modular architecture and real-time functionality make it suitable for interactive uses, thus presenting an accessible, scalable solution to inclusive communication.

---


## Requirements

### Python Dependencies
Install the required Python libraries:
```bash
pip install fastapi keras mediapipe numpy opencv_python pandas sk_video tensorflow uvicorn
```
FFmpeg Installation:

[https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)

---

## To Run the Project

### Clone the repository
```bash
git clone https://github.com/SushmithaS7/Indian-Sign-Language-Recognition.git
cd project
npm install
```

### Start the Frontend

```bash
npm run dev
```

### Start the backend

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

 

