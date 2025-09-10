# Darshini_LF_Person_Detection# Multi-App Analysis Hub: Face Detection & Lost Item Finder

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-orange.svg)
![DeepFace](https://img.shields.io/badge/AI-DeepFace-green.svg)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-blue.svg)

This project is a multi-functional web application hub built with Streamlit. It combines two powerful tools into a single, easy-to-navigate interface:

1.  **Person Detection & Face Analyzer**: An AI-powered tool to analyze facial attributes from an image.
2.  **Lost and Found Item Finder**: A database-driven tool to search for lost items based on their characteristics.

## Features

### 1. Person Detection & Face Analyzer
-   **Facial Analysis**: Upload an image to get an AI-driven estimation of:
    -   Apparent Age Range
    -   Dominant Gender
    -   Dominant Emotion
-   **Find Last Seen Location**: After analysis, search for the same person across a local database of "camera feed" images.
-   **Robust Performance**: Uses the `mtcnn` detector and `Facenet512` model for accurate and reliable face recognition, even with variations in angle and lighting.
-   **Memory Safe**: Designed to handle large models by processing analysis actions sequentially and using garbage collection.

### 2. Lost and Found Item Finder
-   **Database Integration**: Connects securely to a PostgreSQL database to retrieve item information.
-   **Smart Search**: Users can enter 2 or more attributes (e.g., "Laptop", "Dell", "Black") to find matching items.
-   **Scored Results**: The search query scores items based on how many attributes match, showing the most relevant results first.
-   **Dynamic Display**: Presents found items in a clean, responsive grid with images and details.

## Project Structure
```
/LOST_AND_FOUND and AGE DETECTION & FACE ANALYZER/
├── app.py # The main Streamlit router/entry point for the hub.
|
├── Detection_app/ # Module for the Age & Face Detection app.
│ ├── age.py
│ ├── Darshini_logo.png
│
├── lost_and_found/ # Module for the Lost & Found Item Finder app.
│ ├── main.py
│ └── logo.png
│
├── .env # Stores all secret credentials (DATABASE CONNECTION).
├── requirements.txt # A list of all required Python packages.
└── venv/ # The Python virtual environment folder.
```

## Setup and Installation

Follow these steps carefully to get the application running locally.

### 1. Prerequisites
-   Python 3.10 or newer.
-   Git installed on your system.
-   A running PostgreSQL database.

### 2. Clone the Repository
# Project Setup

```bash
# 1. Clone the Repository
git clone <your-repository-url>
cd <your-repository-folder>

# 2. Set Up the Virtual Environment
# Windows:
python -m venv venv
.\venv\Scripts\activate
# macOS / Linux:
python3 -m venv venv
source venv/bin/activate
# You will know it's active when you see (venv) at the beginning of your terminal prompt.

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Configure Environment Variables
# Create a .env file with your PostgreSQL credentials:
echo "DB_USER=\"your_database_user\"
DB_PASSWORD=\"your_database_password\"
DB_HOST=\"localhost\"
DB_PORT=\"5432\"
DB_NAME=\"your_database_name\"" > .env

# 5. Populate Image Database
# Place images into Detection_app/camera_feed_* subfolders
# File names should represent the time, e.g., 10-30-AM.png

# 6. Run the Application
python -m streamlit run app.py
# Your web browser should automatically open with the app running.
# Navigate between "Age Detection" and "Lost and Found" using the sidebar.
