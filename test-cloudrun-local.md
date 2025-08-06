# Testing Cloud Run Environment Locally with Docker

## Setup Instructions

### 1. Start Docker
First, start Docker Desktop on your Mac:
- Open Docker Desktop application
- Wait for it to start (Docker whale icon should be steady in the menu bar)

### 2. Build and Run Container
```bash
# Navigate to your project directory
cd /Users/adipguduru/AutoInterview

# Build the container with Cloud Run simulation (no unnecessary updates)
docker compose build --no-cache

# Start the container with Cloud Run environment variables
docker compose up -d

# Check container logs
docker compose logs -f autointerview
```

### 3. Test the Application

#### Access the Application
- Open your browser and go to: http://localhost:8000
- The app should load with Cloud Run simulation active

#### Verify Cloud Run Environment Detection
Check the container logs for these messages:
```
✓ Cloud Run environment detected
✓ Audio/video functionality disabled for Cloud Run
✓ Video converter disabled in Cloud Run environment
```

#### Test Interview Creation
1. Fill in the role and domain fields
2. Click "Start Interview"
3. Check for successful WebSocket connection in browser dev console
4. Verify no "Cannot Create Interview" error occurs

### 4. What to Look For

#### ✅ Success Indicators:
- Container starts without audio/video library errors
- WebSocket connects using dynamic host (localhost:8000)
- Interview creation succeeds
- Questions are displayed (but audio is simulated)
- Timer functions work properly

#### ❌ Issues to Watch For:
- Missing dependency errors for audio/video libraries
- WebSocket connection failures
- "Cannot Create Interview" errors
- Port binding issues

### 5. Debugging Commands

```bash
# Check container status
docker compose ps

# View real-time logs
docker compose logs -f

# Execute commands inside container
docker compose exec autointerview bash

# Stop and restart
docker compose down
docker compose up -d

# Check environment variables inside container
docker compose exec autointerview env | grep -E "(K_SERVICE|PORT|GOOGLE_CLOUD_PROJECT)"
```

### 6. Test Results to Verify

1. **Environment Detection**: Logs show Cloud Run environment detected
2. **WebSocket Connection**: Browser console shows successful WebSocket connection to ws://localhost:8000/ws/[interview_id]
3. **Interview Creation**: "Start Interview" button works without errors
4. **Graceful Degradation**: Audio/video features are disabled but app still functions
5. **Health Check**: http://localhost:8000/api/health returns success

## Expected Cloud Run Simulation Behavior

- ✅ Interview creation works
- ✅ WebSocket connections establish correctly
- ✅ Questions are displayed as text (audio simulated)
- ✅ Timers function properly
- ✅ No hardware dependency errors
- ✅ Health checks pass
- ✅ No PyAudio/OpenCV/Pygame build errors

## Note: Cloud Run Requirements

This Docker setup uses `requirements-cloudrun.txt` which excludes:
- `pyaudio` (audio recording - not available in Cloud Run)
- `opencv-python` (video capture - not available in Cloud Run)
- `pygame` (audio playback - not available in Cloud Run)
- `streamlit` (not needed for API deployment)

The application gracefully handles missing libraries and operates in "Cloud Run mode" with simulated audio/video functionality.

This local test will validate that your Cloud Run deployment will work correctly!