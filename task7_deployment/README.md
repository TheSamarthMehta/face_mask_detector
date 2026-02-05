# Task 7: Backend Integration & Deployment

## FastAPI Backend

A production-ready REST API for face mask detection with comprehensive documentation and deployment configurations.

### Features

- **FastAPI Framework**: Modern, fast, and well-documented
- **Automatic API Documentation**: Interactive Swagger UI at `/docs`
- **CORS Support**: Cross-origin requests enabled
- **Health Checks**: Built-in monitoring endpoints
- **Error Handling**: Comprehensive error messages
- **Logging**: Structured logging for debugging
- **Production Ready**: Optimized for deployment

## API Endpoints

### 1. Root Endpoint

```
GET /
```

Returns API information and status

### 2. Health Check

```
GET /health
```

Returns server health status

### 3. Prediction

```
POST /predict
```

Upload image for mask detection

**Request:**

- Content-Type: multipart/form-data
- Body: Image file

**Response:**

```json
{
  "success": true,
  "prediction": {
    "class": "With Mask",
    "class_id": 0,
    "confidence": 95.67,
    "color": "#2ecc71"
  },
  "probabilities": {
    "With Mask": 95.67,
    "Without Mask": 2.15,
    "Incorrect Mask": 2.18
  }
}
```

### 4. Get Classes

```
GET /classes
```

Returns available classification classes

### 5. Model Information

```
GET /model-info
```

Returns model architecture details

## Deployment Options

### 1. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python task7_deployment/api.py

# Access API
http://localhost:8000
http://localhost:8000/docs  # Interactive documentation
```

### 2. Docker Deployment

```bash
# Build image
docker build -t face-mask-detection -f task7_deployment/Dockerfile .

# Run container
docker run -p 8000:8000 face-mask-detection

# Or use docker-compose
docker-compose -f task7_deployment/docker-compose.yml up
```

### 3. Railway Deployment

1. Install Railway CLI:

```bash
npm install -g @railway/cli
```

2. Login and deploy:

```bash
railway login
railway init
railway up
```

3. Configuration file: `railway.toml`

### 4. Render Deployment

1. Connect GitHub repository to Render
2. Create new Web Service
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `python task7_deployment/api.py`

### 5. Hugging Face Spaces

1. Create new Space on Hugging Face
2. Upload files:
   - `task7_deployment/api.py`
   - `models/best_model.h5`
   - `requirements.txt`
3. Set SDK to "Docker"
4. Use provided `Dockerfile`

### 6. AWS/GCP/Azure

Deploy using container services:

- AWS: ECS, Fargate, or App Runner
- GCP: Cloud Run or App Engine
- Azure: Container Instances or App Service

## Environment Variables

```bash
MODEL_PATH=models/best_model.h5  # Path to model file
PYTHONUNBUFFERED=1               # Python output buffering
HOST=0.0.0.0                     # Server host
PORT=8000                        # Server port
```

## Production Considerations

### Security

- ✅ Input validation
- ✅ File type checking
- ✅ Size limits
- ✅ Error handling
- ⚠️ Add authentication for production
- ⚠️ Rate limiting recommended

### Performance

- ✅ Model caching
- ✅ Async endpoints
- ✅ Efficient preprocessing
- ⚠️ Consider model quantization for speed
- ⚠️ Add Redis caching for frequent requests

### Monitoring

- ✅ Health check endpoint
- ✅ Structured logging
- ⚠️ Add metrics (Prometheus)
- ⚠️ Add error tracking (Sentry)

## Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction (replace with actual image)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

## Files

- `api.py`: FastAPI application
- `Dockerfile`: Docker configuration
- `docker-compose.yml`: Multi-container setup
- `railway.toml`: Railway deployment config
- `vercel.json`: Vercel deployment config
- `README.md`: Documentation

## API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Load Balancing

For high traffic, use:

- NGINX as reverse proxy
- Multiple worker processes with Gunicorn
- Horizontal scaling with Kubernetes

## Continuous Deployment

Set up CI/CD pipeline:

1. GitHub Actions for automated testing
2. Automatic deployment on push to main
3. Health checks before switching traffic
4. Rollback capability

## Support

For deployment issues:

- Check logs: `docker logs face_mask_detection_api`
- Verify model path
- Ensure all dependencies installed
- Check port availability
