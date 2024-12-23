# MLOps Project: AliTheGreat & Fady Khazzaka

## Project Overview
This project is a collaborative effort by Ali-The-Great and Fady Khazzaka to demonstrate the implementation of MLOps principles in the development and deployment of machine learning models. The focus is on streamlining the lifecycle of ML models, from data preprocessing to monitoring and retraining, using best practices in software engineering for AI.

## Objectives
- **Automate the ML pipeline**: Including data preprocessing, training, evaluation, and deployment.
- **Implement CI/CD**: Ensure consistent and automated testing and deployment of ML models.
- **Scalable and reusable infrastructure**: Develop a framework that can scale to larger datasets and models.
- **Model monitoring and retraining**: Include tools to track model performance and enable retraining as needed.

## Features
- **Data Preprocessing**: Automated data cleaning and feature engineering.
- **Model Training**: Train models with configurable hyperparameters and log performance metrics.
- **Deployment**: Deploy models as RESTful APIs using Docker and Kubernetes.
- **Monitoring**: Real-time monitoring of model performance using tools like Prometheus and Grafana.
- **Version Control**: Git-based tracking of data, models, and code versions.

## Architecture
The project uses the following architecture:

1. **Data Layer**: Raw data is processed and stored in a clean format.
2. **Model Training**: Training pipeline includes hyperparameter tuning and logging via MLFlow.
3. **Deployment**: Models are containerized with Docker and deployed using Kubernetes.
4. **Monitoring**: Logs and metrics are collected and visualized with Prometheus and Grafana.
5. **CI/CD Pipeline**: Implemented using GitHub Actions for automated testing and deployment.

## Prerequisites
- Python 3.9
- Docker
- Kubernetes
- Prometheus and Grafana
- MLFlow
- GitHub Actions
- Required Python libraries (see `requirements.txt`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ali-The-Great/mlops_project_AliTheGreat_Fadykhazzaka.git
   ```
2. Navigate to the project directory:
   ```bash
   cd mlops_project_AliTheGreat_Fadykhazzaka
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up Docker:
   ```bash
   docker-compose up
   ```
5. Deploy Kubernetes cluster:
   ```bash
   kubectl apply -f kubernetes_deployment.yaml
   ```

## Usage
1. **Run the ML pipeline**:
   ```bash
   python pipeline.py
   ```
2. **Monitor the model**: Access Grafana dashboards at `http://localhost:3000`.
3. **Trigger retraining**: Modify configuration files and rerun the pipeline.

## Folder Structure
```
mlops_project_AliTheGreat_Fadykhazzaka/
├── data/
├── models/
├── scripts/
├── docker/
├── kubernetes/
├── requirements.txt
├── pipeline.py
└── README.md
```

## Authors
- **Ali-The-Great**
- **Fady Khazzaka**

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- MLOps course at USJ
- Open-source tools and libraries used in this project
