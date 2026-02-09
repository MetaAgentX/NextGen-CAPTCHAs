#!/bin/bash

# Exit on error
set -e

APP_NAME="captchav2"
REGION="us-central1"  # You can change this to your preferred region

echo "Deploying $APP_NAME to Google Cloud Run..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Check if project is set
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" == "(unset)" ]; then
    echo "Error: No Google Cloud project is set."
    echo "Please run 'gcloud config set project <PROJECT_ID>' to select a project."
    echo "Available projects:"
    gcloud projects list
    exit 1
fi

echo "Deploying $APP_NAME to project $PROJECT_ID in $REGION..."

# Enable APIs
echo "Enabling necessary APIs..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com

# Submit the build to Cloud Build
echo "Building container image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$APP_NAME

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $APP_NAME \
    --image gcr.io/$PROJECT_ID/$APP_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated

echo "Deployment complete!"
