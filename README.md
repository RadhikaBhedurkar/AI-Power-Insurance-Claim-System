# ClaimAI — AI-Assisted Vehicle Insurance Claim Assessment

## 1. Project Overview

**ClaimAI** is an end-to-end AI-powered vehicle insurance claim assessment system designed to assist insurance officers in reviewing vehicle insurance claims.

Traditional insurance claim assessment often requires an employee to manually examine damaged-vehicle photographs, read claim documents, understand the customer's accident description, verify the consistency of the information, and estimate the potential repair cost.

ClaimAI automates and assists with these repetitive tasks by combining **Computer Vision, Natural Language Processing, OCR, Machine Learning, and rule-based validation** into a single workflow.

The system accepts three major inputs:

* Vehicle damage photographs
* Insurance claim documents/PDFs
* Written accident descriptions

These inputs are processed through multiple AI/ML pipelines. The system identifies vehicle damage, estimates damage severity, extracts important information from documents, classifies the accident description, estimates a repair-cost range, and checks whether the information across different sources is consistent.

The final result is presented through a **Streamlit dashboard**, where a human insurance officer can review the evidence, confidence scores, estimated costs, and warning flags before making the final decision.

> **Important:** ClaimAI is a decision-support system. It does not independently approve or reject insurance claims.

---

# 2. Business Problem

Insurance companies receive a large number of vehicle claims containing different types of information.

A typical claim may include:

1. Photographs of the damaged vehicle
2. Insurance policy documents
3. Vehicle registration information
4. Customer information
5. Accident date and location
6. Written accident description
7. Repair information

Manually processing all this information can be:

* Time-consuming
* Repetitive
* Difficult to scale
* Prone to human error
* Inconsistent between reviewers

ClaimAI addresses this problem by creating an **AI-assisted first-level assessment pipeline** that organizes and analyzes the available evidence before it reaches the human reviewer.

---

# 3. End-to-End Workflow

The complete ClaimAI workflow is:

```text
                 USER / CLAIMANT
                       |
                       v
        +-------------------------------+
        |       Claim Submission        |
        |-------------------------------|
        | • Vehicle Photos              |
        | • Claim PDF/Documents         |
        | • Accident Description        |
        +-------------------------------+
                       |
                       v
                FastAPI Backend
                       |
          +------------+------------+
          |            |            |
          v            v            v
    Computer       Document       NLP
      Vision       Intelligence   Analysis
          |            |            |
          v            v            v
   Damage Detection  OCR +        Accident
   + Severity        Entity       Classification
   Classification   Extraction
          |            |            |
          +------------+------------+
                       |
                       v
               Repair Cost Model
                       |
                       v
              Consistency Engine
                       |
                       v
               Assessment Result
                       |
                       v
              Streamlit Dashboard
                       |
                       v
              Human Insurance Officer
                       |
             +---------+---------+
             |         |         |
             v         v         v
          Approve   Reject   Needs Information
```

---

# 4. Input Layer

The system starts with three types of claim evidence.

### A. Vehicle Images

The user uploads photographs of the damaged vehicle.

Examples:

* Front bumper damage
* Scratches
* Dents
* Broken headlights
* Cracked windshield
* Damaged lamps
* Broken glass

These images are passed to the Computer Vision pipeline.

### B. Claim Documents

The user can provide insurance claim documents in PDF format.

The system extracts information such as:

* Policy number
* Vehicle registration number
* Claimant name
* Accident date
* Other available claim information

### C. Accident Description

The claimant provides a written description such as:

> "The vehicle collided with another car at an intersection and the front bumper and headlight were damaged."

This text is analyzed by the NLP pipeline.

---

# 5. FastAPI Backend

The backend is developed using **FastAPI**.

FastAPI acts as the central application layer connecting the frontend with the AI/ML pipelines.

Its responsibilities include:

* Receiving claim information
* Processing uploaded files
* Calling AI/ML models
* Running validation logic
* Generating structured assessment results
* Communicating with the database
* Providing REST API endpoints

The API can be tested using the interactive Swagger documentation:

```text
http://localhost:8000/docs
```

---

# 6. Computer Vision Pipeline

The Computer Vision component analyzes the vehicle photographs.

It consists of two major stages.

## Stage 1 — Damage Detection

A **YOLOv8 object detection model** is used to locate damaged areas in the vehicle image.

The model can identify damage categories such as:

* Dent
* Scratch
* Crack
* Broken glass
* Damaged lamp

The output includes:

* Detected damage type
* Bounding box
* Confidence score
* Location of the detected damage

For example:

```text
Image
   |
   v
YOLOv8
   |
   +---- Dent       91%
   +---- Scratch   87%
   +---- Broken Lamp 94%
```

The damage detection model is designed to be trained using the **CarDD dataset**, subject to the dataset's license and appropriate training/evaluation.

---

# 7. Damage Severity Classification

After identifying damaged areas, the system determines the approximate severity of the damage.

A **ResNet18-based image classification model** is used for severity classification.

The severity categories are:

```text
Minor
Moderate
Severe
```

Example:

```text
Detected Damage:
Front Bumper Dent

Severity:
Moderate

Confidence:
89%
```

This helps the reviewer understand not only what damage was detected but also how serious the detected damage may be.

---

# 8. Document Intelligence Pipeline

Claim documents can contain structured and unstructured information.

ClaimAI uses:

* **pypdf** for extracting text from digital PDFs
* **Tesseract OCR** for extracting text from scanned documents

The extracted text is processed to identify important claim entities.

Example:

```text
Policy Number       → POL-2026-12345
Vehicle Registration → MH05AB1234
Claimant Name       → Example Customer
Accident Date       → 15/09/2026
```

This converts unstructured claim documents into structured information that can be used by the rest of the system.

---

# 9. OCR Processing

For digital PDFs, the system can directly extract available text.

For scanned documents, OCR is required.

The intended workflow is:

```text
Scanned PDF
     |
     v
Page/Image Conversion
     |
     v
Tesseract OCR
     |
     v
Extracted Text
     |
     v
Entity Extraction
```

The current prototype has limitations around scanned PDF rasterization, so this is an area planned for further improvement.

---

# 10. Accident Description Analysis

The written accident description is processed using **Natural Language Processing**.

A Hugging Face transformer-based zero-shot classification approach categorizes the description into classes such as:

* Collision
* Parking damage
* Glass damage
* Theft/Vandalism
* Other

Example:

```text
Input:
"Another vehicle hit my car from behind."

Prediction:
Collision

Confidence:
93%
```

This allows the system to compare the claimant's written description with the damage detected from images.

---

# 11. Repair Cost Estimation

ClaimAI includes a machine learning model to estimate a potential repair-cost range.

Instead of producing a single number, the system is designed to provide a **low-to-high range**.

For example:

```text
Estimated Repair Cost

Low:  ₹12,000
High: ₹22,000
```

The model uses **quantile gradient boosting** to estimate the range.

Potential input features can include:

* Damage type
* Damage severity
* Number of damaged components
* Detected vehicle information
* Other structured claim features

### Important limitation

The current repair-cost model is trained on **synthetic data**.

Therefore, these estimates are illustrative and should **not be treated as real insurance repair estimates**.

For production use, the model would need to be retrained and validated using verified, region-specific repair invoices and actual claims data.

---

# 12. Consistency Checking Engine

One of the important features of ClaimAI is cross-source consistency checking.

The system compares information from:

```text
Images
   +
Accident Description
   +
Claim Documents
```

The goal is to identify potential inconsistencies or missing information.

### Example

The claimant description says:

> "The front bumper was damaged in a collision."

Computer Vision detects:

```text
Rear bumper damage
```

The system can generate a warning:

```text
⚠ Potential mismatch:
Accident description mentions front bumper,
while image analysis indicates rear bumper damage.
```

Other possible flags include:

* Missing policy information
* Missing vehicle registration
* Missing accident date
* Description/damage mismatch
* Unexpected damage category
* Low-confidence predictions

These are **review flags**, not automatic fraud or rejection decisions.

---

# 13. Human-in-the-Loop Review

The final assessment is displayed through a **Streamlit dashboard**.

The dashboard is designed for an insurance officer rather than directly for automatic claim settlement.

The reviewer can see:

### Claim Information

* Claimant
* Policy number
* Vehicle registration
* Accident date

### Image Analysis

* Uploaded images
* Detected damage
* Confidence scores
* Severity

### NLP Analysis

* Accident category
* Classification confidence

### Cost Estimate

```text
Estimated Range:
₹12,000 – ₹22,000
```

### Consistency Flags

```text
✓ Policy information available
✓ Vehicle registration available
⚠ Damage description mismatch
✓ Accident category detected
```

The officer can then record a final review decision such as:

```text
Approve
Reject
Needs Information
```

along with reviewer notes.

---

# 14. Database Layer

ClaimAI uses a database layer for storing claim-related information.

The production-oriented architecture uses:

**PostgreSQL**

For local development, the project can also use:

**SQLite**

The database can store information such as:

* Claim ID
* Claimant information
* Policy information
* Vehicle information
* Uploaded claim information
* Model predictions
* Confidence scores
* Cost estimates
* Consistency flags
* Reviewer decision
* Reviewer notes

---

# 15. Technology Architecture

The major technologies used are:

### Backend

* Python
* FastAPI
* SQLAlchemy

### Computer Vision

* PyTorch
* YOLOv8
* torchvision
* OpenCV
* ResNet18

### NLP

* Hugging Face Transformers
* Zero-shot classification

### Document AI

* pypdf
* Tesseract OCR

### Machine Learning

* scikit-learn
* Gradient Boosting
* Quantile regression

### Frontend

* Streamlit

### Database

* PostgreSQL
* SQLite for local development

### Deployment

* Docker
* Docker Compose

---

# 16. Project Architecture

```text
                    CLAIM SUBMISSION
                           |
           +---------------+---------------+
           |               |               |
           v               v               v
       Images           Documents      Description
           |               |               |
           v               v               v
        YOLOv8          PDF/OCR          NLP
           |               |               |
           v               v               v
    Damage Detection   Entity Extraction  Accident
           |               |              Category
           v               |               |
       ResNet18             |               |
           |                |               |
           v                v               v
      Severity       Structured Data    Classification
           |                |               |
           +----------------+---------------+
                            |
                            v
                    Feature Processing
                            |
                            v
                    Repair Cost Model
                            |
                            v
                 Consistency Check Engine
                            |
                            v
                     Claim Assessment
                            |
                            v
                  Streamlit Dashboard
                            |
                            v
                  Human Insurance Officer
                            |
              +-------------+-------------+
              |             |             |
           APPROVE        REJECT      NEEDS INFO
```

---

# 17. Demo Mode

The project currently supports **Demo Mode**.

Demo Mode allows the complete application workflow to be demonstrated without requiring trained production models.

It can be started using:

```powershell
$env:DEMO_MODE="1"
uvicorn app.api:app --reload
```

The demo mode generates placeholder results based on available inputs and clearly identifies the outputs as:

```text
DEMO MODE
```

This makes it possible to demonstrate the application architecture before the real models are trained.

---

# 18. Model Training Pipeline

The project separates application code from model training.

Training scripts include:

```text
training/
├── train_yolo.py
├── train_severity.py
└── train_cost.py
```

The intended workflow is:

```text
Dataset
   |
   v
Data Cleaning
   |
   v
Preprocessing
   |
   v
Train / Validation / Test Split
   |
   v
Model Training
   |
   v
Model Evaluation
   |
   v
Save Trained Weights
   |
   v
Integrate with FastAPI
   |
   v
Production Inference
```

---

# 19. Model Evaluation

For a production-ready version, each component should have dedicated evaluation metrics.

### YOLOv8

* Precision
* Recall
* mAP@50
* mAP@50:95

### Severity Classifier

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

### NLP Classifier

* Accuracy
* Precision
* Recall
* F1-score

### Cost Model

* MAE
* RMSE
* Quantile coverage
* Prediction interval quality

### OCR

* Character-level accuracy
* Entity extraction accuracy

These metrics would allow the system's performance to be objectively evaluated.

---

# 20. Docker Architecture

The project can be containerized using Docker Compose.

The planned deployment contains:

```text
Docker Compose
      |
      +---- PostgreSQL
      |
      +---- FastAPI
      |
      +---- Streamlit
```

The complete application can therefore be started using:

```bash
docker compose up --build
```

FastAPI runs on:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

Streamlit dashboard:

```text
http://localhost:8501
```

---

# 21. Complete Example

Suppose a customer submits:

### Vehicle Images

The images show a damaged front bumper and broken headlight.

### Accident Description

> "My car collided with another vehicle at a traffic signal and the front bumper and headlight were damaged."

### Insurance Document

Contains:

```text
Policy Number: POL12345
Vehicle Number: MH05AB1234
Claimant: Radhika Bhedurkar
Accident Date: 15/09/2026
```

ClaimAI processes the information.

### Computer Vision

```text
Front bumper damage → detected
Broken headlight → detected
```

### Severity

```text
Moderate
```

### NLP

```text
Accident Type → Collision
```

### Document Intelligence

```text
Policy Number → extracted
Vehicle Number → extracted
Claimant → extracted
Accident Date → extracted
```

### Cost Model

```text
Estimated repair range:
₹15,000 – ₹28,000
```

### Consistency Engine

```text
✓ Description matches detected damage
✓ Vehicle information available
✓ Accident date available
✓ Policy number available
```

### Dashboard

The insurance officer sees the complete evidence and can select:

```text
Approve
Reject
Needs Information
```

The AI provides supporting evidence, while the **human reviewer remains responsible for the final decision**.

---

# 22. Current Limitations

The prototype currently has several limitations:

1. Repair-cost predictions are based on synthetic data.
2. Real-world model performance depends on properly trained and evaluated datasets.
3. Scanned PDFs require additional rasterization/OCR processing.
4. Multimodal fusion is currently rule-based.
5. No authentication or role-based access is implemented yet.
6. No production fraud-detection model is currently implemented.
7. Dataset licenses must be reviewed before commercial use.
8. Demo Mode uses placeholder logic rather than production-trained models.

---

# 23. Future Improvements

The roadmap includes:

### Advanced NLP

Fine-tune a **DistilBERT-based accident classifier** using insurance-specific data.

### Vision-Language Model

Introduce a vision-language model to directly compare:

```text
Image ↔ Accident Description ↔ Claim Document
```

for more advanced multimodal reasoning.

### Fraud Risk Detection

Add a separate fraud-risk scoring model to identify claims requiring additional investigation.

### Audit Trail

Maintain a complete record of:

* Model predictions
* Reviewer actions
* Changes
* Evidence
* Final decision

### Authentication

Implement:

* User authentication
* Role-based access
* Insurance officer accounts
* Administrator accounts

### Mobile Application

Develop a Flutter or React Native client using the same FastAPI backend.

### Production Model Evaluation

Create automated evaluation reports covering:

* Damage detection mAP
* Severity F1
* NLP F1
* Cost MAE
* Calibration/confidence
* Error analysis

---

# 24. Why This Is an End-to-End AI Project

ClaimAI demonstrates more than simply training an ML model.

It covers the complete AI application lifecycle:

```text
Business Problem
       ↓
Data Collection
       ↓
Data Processing
       ↓
Computer Vision
       ↓
NLP
       ↓
OCR
       ↓
Machine Learning
       ↓
Model Inference
       ↓
Rule-Based Validation
       ↓
Backend API
       ↓
Database
       ↓
Web Dashboard
       ↓
Human Review
       ↓
Docker Deployment
```

👩‍💻 Author

Radhika Bhedurkar

Data Analyst & Data Science Enthusiast

Connect with me LinkedIn: https://www.linkedin.com/in/radhika-bhedurkar

GitHub: https://github.com/RadhikaBhedurkar

