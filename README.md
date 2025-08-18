# HumanBrand AI - Brand Analysis API

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103.2-blueviolet)](https://fastapi.tiangolo.com/)

An asynchronous API for deep brand analysis of large document sets using AWS Bedrock, intelligent chunking, and a multi-agent AI system.

## ✨ Key Features

-   **Asynchronous API**: Built with FastAPI for handling long-running AI analysis jobs.
-   **AI-Powered Analysis**: Leverages AWS Bedrock (Claude 3.7 Sonnet) for state-of-the-art language understanding.
-   **Smart Document Chunking**: Intelligently processes large documents without context loss.
-   **Intelligent Routing**: Automatically chooses between single-pass or batch processing.
-   **Real-Time Progress**: WebSocket support for live status updates.
-   **Structured Output**: Delivers results as a markdown report and structured JSON.

## 🚀 Getting Started

### 1. Prerequisites

-   Python 3.9+
-   An AWS account with Bedrock access.

### 2. Setup

```bash
# Clone the repository
git clone <your-repository-url>
cd <your-repository-directory>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# .\venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
3. Configure Environment
Create a .env file in the project root and add your AWS credentials:
code
Dotenv
# .env
AWS_ACCESS_KEY_ID=AKIAxxxxxxxxxxxxxxxx
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_REGION=eu-north-1
4. Run the Server
code
Bash
uvicorn fastapi_backend:app --reload
The API will be running at http://127.0.0.1:8000.
📖 API Workflow
Interact with the API via the auto-generated docs at http://127.0.0.1:8000/docs.
POST /api/v1/upload-files
Upload your corpus files.
Copy the upload_id from the successful response.
POST /api/v1/start-analysis
Use the upload_id from the previous step in the request body.
Copy the job_id from the successful response.
GET /api/v1/analysis/{job_id}
Use the job_id to poll for the analysis status and progress.
GET /api/v1/results/{job_id}
Once the job status is completed, use the job_id to retrieve the final analysis report and data.
Endpoints Quick Reference
Method	Endpoint	Description
POST	/api/v1/upload-files	Upload files. Returns an upload_id.
POST	/api/v1/start-analysis	Start analysis job. Returns a job_id.
GET	/api/v1/analysis/{job_id}	Get the step-by-step status of a job.
GET	/api/v1/results/{job_id}	Get complete results for a finished job.
GET	/api/v1/download/{job_id}/{type}	Download results (markdown, json, or full).
WS	/ws/analysis/{job_id}	WebSocket for real-time progress updates.
GET	/health	System health check.
