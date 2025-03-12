# California Housing Price Prediction System

🔗 **Live Demo:** [House Price Predictor](http://ec2-13-203-104-133.ap-south-1.compute.amazonaws.com)

📄 **Project Report:** [Full Documentation](https://lijoraju.github.io/house-price-predictor/#conclusion)

## Project Overview
This project builds and deploys a machine learning model to predict house prices in California based on various housing and demographic attributes. The system includes:
- A **Flask API** for model serving.
- A **machine learning model** trained on the California Housing Dataset.
- A **frontend web interface** for user interaction.
- **Deployment on AWS** using Docker.

## Features
- **Data Preprocessing and Feature Engineering**
- **Multiple Model Training & Hyperparameter Tuning**
- **Model Selection & Performance Evaluation**
- **REST API for Predictions**
- **Interactive Frontend for User Input**
- **Logging and Error Handling** for better debugging and monitoring.
- **MLflow for Model Versioning** to track and manage different model versions.
- **Live Deployment on AWS**

## Installation & Setup
### Prerequisites
Ensure you have Python 3.9+ installed on your system. You also need `pip` for package management and `Docker` if deploying in a containerized environment.

### Clone the Repository
```sh
git clone https://github.com/yourusername/california-housing-prediction.git
cd california-housing-prediction
```

### Install Dependencies
Install all required packages using:
```sh
pip install -r requirements.txt
```

### Run the Flask API Locally
```sh
python app.py --port=8000
```
The API will be available at `http://127.0.0.1:8000/predict`.

### Example API Request (cURL)
```sh
curl -X POST -H "Content-Type: application/json" -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.02381,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
}' http://127.0.0.1:8000/predict
```

### Run the Frontend
Simply open `index.html` in your browser or deploy it on a simple web server.

## Deployment
### Containerization with Docker
To build and run the application as a Docker container:
```sh
docker build -t housing-price-api .
docker run -p 8000:8000 housing-price-api
```

### Deployment on AWS
The API is deployed on AWS and accessible at:
```sh
http://ec2-13-203-104-133.ap-south-1.compute.amazonaws.com/predict
```

## Technologies Used
- **Flask** - API development
- **XGBoost, Scikit-learn** - Machine Learning
- **MLflow** - Model tracking
- **Docker** - Containerization
- **AWS EC2** - Cloud deployment
- **HTML, JavaScript** - Frontend UI

## Contributing
Feel free to open issues or pull requests to improve this project!

## License
This project is licensed under the MIT License.
