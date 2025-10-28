# 🎓 Master's Thesis – Sail Line Detection Using SegFormer Image Segmentation by Henrique Teixeira

## 🧭 Context
This repository contains the **source code** developed as part of my **Master’s Thesis in Critical Computing Engineering** at the **Instituto Superior de Engenharia do Porto (ISEP)**.  
The project focuses on **detecting sail lines** in images of sailboats to determine **aerodynamic parameters** such as **Camber**, **Draft**, and **Twist**, using **Computer Vision** and **Artificial Intelligence** techniques.

My thesis document is included in this directory as a PDF file.

---

## 📖 Project Overview

The purpose of this project is to design and implement an **intelligent and efficient system** that can:
- Detect and track the **orange lines** placed on a sail using a GoPro camera.  
- Automatically determine **Camber**, **Draft**, and **Twist** from the detected lines.  
- Operate efficiently in **near real-time** on **low-power devices** such as a GoPro and tablet.  

This project combines **computer vision** with **deep learning** to achieve high-accuracy line detection with minimal computational cost.  It makes use of **SegFormer**, a transformer-based semantic segmentation model, to segment and identify the sail lines.

> ⚠️ This repository includes **only the source code** used in the development of the project.  
> The dataset, videos, trained models, and extracted parameters are **not** included here.  
> The dataset used for the training of the model is available on **Kaggle** in the link below.

🔗 [https://www.kaggle.com/datasets/henriqueteixeira22/sail-dataset](X) 

---

## Install Dependencies
Make sure you have Python 3.10+ installed.
Then install the required packages using 

> pip install -r requirements.txt

## (Optional) Install CUDA
If you have an NVIDIA GPU, install the CUDA Toolkit for faster model training and inference using the video I used:
> 🔗 [https://youtu.be/uL3vXRI-7rs?list=LL](X)

or

> https://developer.nvidia.com/cuda-downloads

---

## Dataset Setup

1. Download the dataset from Kaggle:  
   **Sail Line Detection Dataset – Kaggle** → 🔗 [https://www.kaggle.com/datasets/henriqueteixeira22/sail-dataset](X)

2. After downloading, extract the dataset and place it inside the project folder named "trainingmodel".
3. The training and testing scripts read data from `trainingmodel/` once you set the variables inside each script.

---

## 🧠 Training a Model

This project supports three SegFormer variants: **MiT-B1**, **MiT-B2** and **MiT-B3** 

### Steps to Train
1. Open the training script:
2. Edit the variables at the top of the file:
3. Run the script with
    > python segformer.py

4. Once training is complete, the trained weights will be saved in the same trainingmodel/ folder. The script also includes a checkpoint mechanism, allowing you to stop the training process at any time and later resume it from the same epoch where it left off.

---

## 🎥 Testing with a Sail Video

After training your model, you can test it on a sail video to visualize detection and the aerodynamic parameters analysis.

### Steps to Test
1. Place your **test video** in the "teseHT/screenshotsdosvideos/videos" folder in this format: **Video X.MP4**.  
2. Open the file "chordpoints.py":
3. Edit the variables:
- **`VIDEO_PATH`** → The path to the input video file you want to analyze.  
     Example: `"teseHT/screenshotsdosvideos/videos/Video 1.MP4"`

- **`ARCH`** → The model architecture you want to use for testing.  
     Choose between `"mit_b1"`, `"mit_b2"`, or `"mit_b3"` depending on which SegFormer model you trained.

 - **`CHECKPOINT`** → The path to the saved model weights (.pth file) generated after training.  
     Example: `"trainingmodel/segformer_mitb1.pth"`

 - **`ROOT_DIR`** → The root directory where the dataset and results are stored.  
     Example: `"teseHT/.../trainingmodel/sail_dataset"`

4. Run the script using
> python chordpoints.py

5. The program will:
    - Process the video frame by frame
    - Detect and highlight sail lines
    - Display the result and/or save the processed output video to the chosen path

## 🧩 Technologies Used

- **Python 3.10+**.
- **PyTorch** — deep learning framework for training and inference  
- **SegFormer (MiT-B1/B2/B3)** — transformer-based semantic segmentation models  
- **OpenCV** — video and image processing  
- **NumPy** — numerical computations  
- **Matplotlib** — visualization  
- **Scikit-learn** — metrics/utilities  
- **CUDA (optional)** — GPU acceleration  

## ⚙️ How It Works (Simplified)

1. **Data Preparation**  
   Sail images and masks are loaded from `trainingmodel/`, preprocessed, and fed to the model.
2. **Model Training**  
   The selected SegFormer model learns to segment sail lines from labeled data.
3. **Inference**  
   The trained model processes new images or videos to produce segmentation masks of the sail lines.
4. **Visualization & (Optional) Parameter Extraction**  
   Segmented lines are overlaid for visualization and can be used to compute aerodynamic parameters such as **Camber**, **Draft**, and **Twist**.

## 👤 Author
**Henrique Manuel de Almeida e Silva dos Santos Teixeira**  
Master’s in Critical Computing Systems Engineering  
Instituto Superior de Engenharia do Porto (ISEP)  

📧 Email: [henriqueteixeira227@gmail.com]  
🔗 LinkedIn: [https://www.linkedin.com/in/henrique-teixeira-032562261](https://www.linkedin.com/in/henrique-teixeira-032562261)

