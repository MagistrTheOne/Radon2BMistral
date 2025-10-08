#!/bin/bash
"""
RADON AWS Quick Start Script
Deploy and train RADON models on AWS
"""

set -e

# Configuration
REGION="us-east-1"
BUCKET="radon-models-$(date +%s)"
ROLE="arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole"

echo "🚀 RADON AWS Quick Start"
echo "Region: $REGION"
echo "Bucket: $BUCKET"

# Create S3 bucket
echo "📦 Creating S3 bucket..."
aws s3 mb s3://$BUCKET --region $REGION

# Upload code and configs
echo "📤 Uploading RADON code..."
aws s3 cp . s3://$BUCKET/code/ --recursive --exclude "*.git*" --exclude "*.pyc"

# Upload model configs
echo "📤 Uploading model configs..."
aws s3 cp configs/ s3://$BUCKET/configs/ --recursive

# Create training datasets (sample)
echo "📊 Creating sample datasets..."
mkdir -p /tmp/datasets
cat > /tmp/datasets/russian.json << EOF
[
    {"text": "Привет! Как дела? Это тестовый текст на русском языке."},
    {"text": "Сегодня хорошая погода. Я иду в парк."},
    {"text": "Машинное обучение - это интересная область."}
]
EOF

cat > /tmp/datasets/english.json << EOF
[
    {"text": "Hello! How are you? This is a test text in English."},
    {"text": "Today is a beautiful day. I'm going to the park."},
    {"text": "Machine learning is an interesting field."}
]
EOF

cat > /tmp/datasets/code.json << EOF
[
    {"text": "def hello_world():\n    print('Hello, World!')"},
    {"text": "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()"},
    {"text": "def train_model(model, data):\n    optimizer = torch.optim.Adam(model.parameters())\n    for epoch in range(10):\n        loss = model(data)\n        loss.backward()\n        optimizer.step()"}
]
EOF

# Upload datasets
aws s3 cp /tmp/datasets/ s3://$BUCKET/data/training/ --recursive

# Build and push Docker image
echo "🐳 Building Docker image..."
docker build -f aws/docker/Dockerfile.aws -t radon-training:latest .

# Tag for ECR
ECR_REPO="radon-training"
aws ecr create-repository --repository-name $ECR_REPO --region $REGION || true
ECR_URI=$(aws ecr describe-repositories --repository-names $ECR_REPO --region $REGION --query 'repositories[0].repositoryUri' --output text)

# Login to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI

# Tag and push
docker tag radon-training:latest $ECR_URI:latest
docker push $ECR_URI:latest

# Create SageMaker training job
echo "🏋️ Starting SageMaker training job..."

# Training configuration
cat > /tmp/training_config.json << EOF
{
    "TrainingJobName": "radon-training-$(date +%s)",
    "RoleArn": "$ROLE",
    "AlgorithmSpecification": {
        "TrainingImage": "$ECR_URI:latest",
        "TrainingInputMode": "File"
    },
    "ResourceConfig": {
        "InstanceType": "ml.p3.2xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 50
    },
    "InputDataConfig": [
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://$BUCKET/data/training/",
                    "S3DataDistributionType": "FullyReplicated"
                }
            }
        }
    ],
    "OutputDataConfig": {
        "S3OutputPath": "s3://$BUCKET/models/"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 3600
    }
}
EOF

# Start training job
aws sagemaker create-training-job --cli-input-json file:///tmp/training_config.json --region $REGION

echo "✅ Training job started!"
echo "Monitor progress: aws sagemaker describe-training-job --training-job-name radon-training-* --region $REGION"

# Cleanup
rm -rf /tmp/datasets /tmp/training_config.json

echo "🎉 RADON AWS setup complete!"
echo "Bucket: s3://$BUCKET"
echo "ECR Repository: $ECR_URI"
