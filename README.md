# AutoInterview - AI-Powered Video Interview System

An automated interview system that conducts structured technical interviews using AI, featuring real-time video recording, speech-to-text transcription, and comprehensive candidate evaluation.

## Features

- **Structured Interview Process**: Follows a predefined interview structure with timed responses
- **Video Recording**: Continuous video recording with real-time audio extraction
- **Speech-to-Text**: Real-time transcription using Google Gemini AI
- **Multi-Modal Input**: Supports both audio-only and video modes
- **Real-Time Updates**: WebSocket integration for live interview progress
- **Automated Scoring**: Comprehensive evaluation with performance metrics
- **RESTful API**: FastAPI backend for easy integration

## Interview Structure

1. **Greeting** - Welcome message
2. **Self Introduction** (30 seconds)
3. **Experience Inquiry** (45 seconds)
4. **True/False Questions** (2 × 10 seconds)
5. **Multiple Choice Questions** (2 × 15 seconds)
6. **Theoretical Questions** (2 × 30 seconds)
7. **Situational Questions** (2 × 30 seconds)
8. **Candidate Questions** (60 seconds)
9. **Closing** - Thank you message

## Prerequisites

- Python 3.8+
- Google Gemini API key
- Camera and microphone (for video mode)
- Audio drivers (PyAudio compatible)

### System Requirements

- **macOS**: Built-in 'say' command (no additional installation)
- **Linux**: Install `espeak` or `festival` (`sudo apt-get install espeak`)
- **Windows**: Requires `pywin32` for SAPI voice support

## Installation

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AdipG-ttec/AutoInterview.git
   cd AutoInterview
   ```

2. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

3. **Run with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

   The application will be available at `http://localhost:8000`

4. **View logs:**
   ```bash
   docker-compose logs -f autointerview
   ```

5. **Stop the application:**
   ```bash
   docker-compose down
   ```

### Option 2: Manual Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AdipG-ttec/AutoInterview.git
   cd AutoInterview
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Usage

### Running the FastAPI Server

```bash
python run_api.py
```

The server will start on `http://localhost:8000`

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Running Direct Interview Mode

```bash
python server.py
```

Choose between:
- **Audio-only mode** (microphone input)
- **Video mode** (continuous recording + audio extraction)

## API Endpoints

### Create Interview
```http
POST /api/interviews/create
Content-Type: application/json

{
  "role_name": "Software Engineer",
  "domain": "Machine Learning",
  "camera_index": 0,
  "use_video": true
}
```

### Start Interview
```http
POST /api/interviews/{interview_id}/start
```

### Get Interview Status
```http
GET /api/interviews/{interview_id}/status
```

### Stop Interview
```http
POST /api/interviews/{interview_id}/stop
```

### WebSocket Connection
```
ws://localhost:8000/ws/{interview_id}
```

## Configuration

### Interview Configuration
- `role_name`: Position being interviewed for
- `domain`: Technical domain/field
- `camera_index`: Camera device index (default: 0)
- `use_video`: Enable video recording (default: true)

### Audio Settings
- Sample Rate: 24kHz
- Channels: Mono
- Format: 16-bit PCM

### Video Settings
- Resolution: 16:9 aspect ratio (auto-detected)
- Frame Rate: 30 FPS
- Codec: MP4V (with fallback options)

## File Structure

```
AutoInterview/
├── server.py              # Main interview system
├── run_api.py             # FastAPI server launcher
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in git)
├── .gitignore            # Git ignore rules
├── app.js                # Frontend JavaScript
├── index.html            # Frontend interface
├── style.css             # Frontend styling
└── README.md             # This file
```

## Output Files

The system generates several output files:

- `interview_video_YYYYMMDD_HHMMSS.mp4` - Complete video recording
- `interview_audio_YYYYMMDD_HHMMSS.wav` - Audio track
- `structured_interview_report_ROLE_YYYYMMDD_HHMMSS.json` - Evaluation report

## Evaluation Metrics

The system provides scoring in four areas:

1. **Technical Knowledge** - Based on T/F and MCQ correctness
2. **Theoretical Understanding** - Quality of theoretical responses
3. **Problem-Solving Ability** - Situational question responses
4. **Communication Skills** - Clarity and time management

## Security

- API keys stored in environment variables
- Sensitive files excluded via `.gitignore`
- No hardcoded credentials in source code

## Docker Commands

### Building and Running

```bash
# Build the Docker image
docker build -t autointerview .

# Run with Docker Compose (recommended)
docker-compose up -d

# Run container manually
docker run -d \
  --name autointerview \
  -p 8000:8000 \
  -e GEMINI_API_KEY=your_api_key_here \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/audio:/app/audio \
  -v $(pwd)/reports:/app/reports \
  --device /dev/snd:/dev/snd \
  autointerview
```

### Management Commands

```bash
# View logs
docker-compose logs -f autointerview

# Access container shell
docker-compose exec autointerview bash

# Restart application
docker-compose restart autointerview

# Update and rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Volume Mounts

The Docker setup includes persistent volumes for:
- `/app/videos` - Video recordings
- `/app/audio` - Audio files
- `/app/reports` - Interview reports

## Troubleshooting

### Common Issues

1. **Audio not working**: Check PyAudio installation and microphone permissions
2. **Video not recording**: Verify camera permissions and OpenCV installation
3. **API key errors**: Ensure `.env` file exists with valid `GEMINI_API_KEY`
4. **Import errors**: Install all requirements with `pip install -r requirements.txt`

### Docker-Specific Issues

1. **Audio/Video in Docker**: 
   - Ensure proper device permissions: `--device /dev/snd:/dev/snd`
   - For video: `--device /dev/video0:/dev/video0`
   - May require additional host configuration

2. **Permissions**: Run with appropriate user permissions:
   ```bash
   docker-compose exec autointerview chown -R $(id -u):$(id -g) /app/videos /app/audio /app/reports
   ```

### Debug Mode

Set environment variable for verbose logging:
```bash
# Local installation
export PYTHONPATH=.
python -u server.py

# Docker
docker-compose exec autointerview python -u server.py
```

## License

This project is private and proprietary.

## Support

For issues and questions, please create an issue in the GitHub repository.