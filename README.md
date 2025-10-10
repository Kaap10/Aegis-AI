

# AegisAI 🛡️

An AI-powered cybersecurity platform designed as an "immune system" for critical digital infrastructure. AegisAI provides a unified defense layer by combining AI model security with real-time cloud and identity anomaly detection.

## 📖 Project Description

Modern systems increasingly rely on AI models and cloud infrastructure, creating new vulnerabilities. AI models can be tricked by adversarial attacks, leading to catastrophic failures, while cloud identity threats like brute-force attacks and impossible travel are constant risks. Traditional security solutions often operate in silos, leading to slow and inefficient responses.

AegisAI tackles this challenge by providing an integrated platform that protects the entire ecosystem. It hardens AI models to make them resilient to attacks and simultaneously monitors user and cloud activity to detect and flag threats in real-time, all demonstrated through an interactive dashboard.

-----

## ✨ Key Features

  * **AI Model Security:** Utilizes **Adversarial Training** to harden PyTorch models (ResNet-18) against attacks like FGSM and PGD, significantly improving their robustness and reliability.
  * **Cloud & Identity Monitoring:** Employs an `IsolationForest` model to perform real-time anomaly detection on simulated user authentication and cloud logs, successfully identifying threats like "impossible travel" and brute-force activity.
  * **Interactive Dashboard:** A user-friendly frontend built with **Streamlit** that provides a live demo of both the model security and anomaly detection features.
  * **Containerized Deployment:** The entire multi-service application (FastAPI backend + Streamlit frontend) is containerized with **Docker**, allowing for a simple, one-command launch on any system.

-----

## 🚀 Live Demo / Screenshots

Here's a look at the AegisAI dashboard in action.
![AegisAI Dashboard](/Assets/main.png)

#### **AI Model Security Simulation**

The dashboard allows you to upload an image and see how the hardened model correctly identifies it, even after being subjected to an adversarial attack that fools a standard model.

![Simulation](/Assets/adversarial_attack.png)


#### **Cloud & Identity Monitoring Feed**

The simulation feed shows events as they happen, with the AI correctly distinguishing between normal (🟢) and anomalous (🔴) activities in real-time.

![Monitoring](/Assets/anamoly_detection.png)
-----

## 💻 Tech Stack

| Component | Technology / Library |
| :--- | :--- |
| **Backend** | Python, FastAPI |
| **Frontend** | Streamlit |
| **ML / AI** | PyTorch, scikit-learn, Adversarial Robustness Toolbox (ART), Pandas |
| **DevOps** | Docker, Docker Compose |

-----

## 🛠️ Setup & Run Locally

This project is fully containerized, making the setup process simple and consistent.

#### **Prerequisites**

  * [Git](https://git-scm.com/downloads)
  * [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

#### **Running the Application**

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Kaap10/Aegis-AI.git
    cd Aegis-AI
    ```

2.  **Launch with Docker Compose:**
    From the project's root directory, run the following command. This will build the images and start both the backend and frontend services.

    ```bash
    docker-compose up --build
    ```

    The initial build may take several minutes as it downloads and installs all dependencies.

3.  **Access the Dashboard:**
    Once the containers are running, open your web browser and navigate to:
    **`http://localhost:8501`**

4.  **Stopping the Application:**
    To stop and remove the containers, press `Ctrl+C` in the terminal where the application is running, and then run:

    ```bash
    docker-compose down
    ```

-----

## 📂 Project Structure

```
Aegis-AI/
├── .dockerignore
├── data/
├── models/
│   ├── anomaly_detector.pkl
│   └── hardened_model.pt
├── src/
│   ├── api/
│   │   ├── Dockerfile
│   │   └── main.py
│   └── dashboard/
│       ├── Dockerfile
│       └── dashboard.py
├── docker-compose.yml
├── requirements.txt
└── README.md
```