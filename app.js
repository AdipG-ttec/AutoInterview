class AutoInterviewApp {
    constructor() {
        this.apiBase = 'localhost:8000'; // Default API base URL, can be overridden by environment
        this.interviewId = null;
        this.websocket = null;
        this.videoStream = null;
        this.timerInterval = null;
        this.currentStage = null;
        this.timeRemaining = 0;
        this.totalTime = 0;
        this.isStarted = false;
        this.isLoading = false;
        
        this.initializeElements();
        this.setupEventListeners();
        this.initializeApp();
    }

    initializeElements() {
        this.elements = {
            // Start screen elements
            startScreen: document.getElementById('start-screen'),
            startBtn: document.getElementById('start-btn'),
            startRole: document.getElementById('start-role'),
            startDomain: document.getElementById('start-domain'),
            
            // Main app elements
            appContainer: document.getElementById('app-container'),
            videoFeed: document.getElementById('video-feed'),
            videoPlaceholder: document.getElementById('video-placeholder'),
            questionText: document.getElementById('question-text'),
            currentStage: document.getElementById('current-stage'),
            stageIndicator: document.getElementById('stage-indicator'),
            timeBar: document.getElementById('time-bar'),
            timeBarContainer: document.getElementById('time-bar-container'),
            timeLabel: document.getElementById('time-label'),
            
            // Status and info elements
            statusSection: document.getElementById('status-section'),
            statusMessage: document.getElementById('status-message'),
            roleName: document.getElementById('role-name'),
            domainName: document.getElementById('domain-name')
        };
        
        // Debug: Check if all elements exist
        console.log('Elements check:');
        for (const [key, element] of Object.entries(this.elements)) {
            if (!element) {
                console.error(`Missing element: ${key}`);
            } else {
                console.log(`✓ Found element: ${key}`);
            }
        }
    }

    setupEventListeners() {
        this.elements.startBtn.addEventListener('click', () => this.handleStart());
        
        // Handle browser tab close/refresh
        window.addEventListener('beforeunload', () => {
            if (this.interviewId) {
                this.cleanup();
            }
        });
    }

    async initializeApp() {
        try {
            console.log('Initializing app...');
            // Check if API is available
            const response = await fetch(`${this.apiBase}/api/health`);
            console.log('Health check response:', response.status);
            if (!response.ok) {
                throw new Error('API server not available');
            }
            
            const healthData = await response.json();
            console.log('Health check data:', healthData);
            console.log('App initialized successfully');
            
        } catch (error) {
            console.error('Initialization error:', error);
            this.showStatus('Failed to initialize. Please check if the API server is running.', 'error');
            this.elements.startBtn.disabled = true;
        }
    }

    async initializeVideo() {
        try {
            this.videoStream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    width: { ideal: 1920 },
                    height: { ideal: 1080 },
                    aspectRatio: 16/9 
                },
                audio: true
            });
            
            this.elements.videoFeed.srcObject = this.videoStream;
            this.elements.videoPlaceholder.style.display = 'none';
            
            return true;
        } catch (error) {
            console.error('Video initialization error:', error);
            this.showStatus('Camera access denied. Please allow camera permissions and refresh the page.', 'error');
            return false;
        }
    }

    async handleStart() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.elements.startBtn.disabled = true;
        this.elements.startBtn.textContent = 'Preparing Interview...';
        
        try {
            // Start camera
            const videoSuccess = await this.initializeVideo();
            if (!videoSuccess) {
                throw new Error('Camera access required');
            }
            
            console.log('Camera initialized successfully');
            
            // Create interview
            console.log('Creating interview...');
            const interviewId = await this.createInterview();
            this.interviewId = interviewId;
            console.log('Interview created:', interviewId);
            
            // Transition to main app first
            this.transitionToMainApp();
            
            // Connect WebSocket
            console.log('Connecting WebSocket...');
            this.connectWebSocket(interviewId);
            
            // Wait for WebSocket to connect, then start interview
            this.waitForWebSocketConnection().then(async () => {
                try {
                    console.log('WebSocket connected, starting interview...');
                    await this.startInterview(interviewId);
                    console.log('Interview started successfully');
                } catch (error) {
                    console.error('Error starting interview:', error);
                    this.showStatus(`Failed to start interview: ${error.message}`, 'error');
                }
            }).catch((error) => {
                console.error('WebSocket connection failed:', error);
                this.showStatus('Failed to establish WebSocket connection', 'error');
            });
            
        } catch (error) {
            console.error('Start interview error:', error);
            this.showStatus(`Failed to start interview: ${error.message}`, 'error');
            this.elements.startBtn.disabled = false;
            this.elements.startBtn.textContent = 'Start Interview';
        } finally {
            this.isLoading = false;
        }
    }

    async createInterview() {
        try {
            console.log('Sending create interview request...');
            const response = await fetch(`${this.apiBase}/api/interviews/create`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    role_name: this.elements.roleName.textContent,
                    domain: this.elements.domainName.textContent,
                    camera_index: 0,
                    use_video: true
                })
            });

            console.log('Create interview response status:', response.status);
            const data = await response.json();
            console.log('Create interview response data:', data);
            
            if (data.success) {
                return data.interview_id;
            }
            throw new Error(data.message || 'Failed to create interview');
        } catch (error) {
            console.error('Create interview error:', error);
            throw error;
        }
    }

    async startInterview(id) {
        try {
            console.log('Sending start interview request for:', id);
            const response = await fetch(`${this.apiBase}/api/interviews/${id}/start`, {
                method: 'POST'
            });

            console.log('Start interview response status:', response.status);
            const data = await response.json();
            console.log('Start interview response data:', data);
            
            if (!data.success) {
                throw new Error(data.message || 'Failed to start interview');
            }
        } catch (error) {
            console.error('Start interview error:', error);
            throw error;
        }
    }

    connectWebSocket(id) {
        // Use dynamic host and protocol detection for Cloud Run compatibility
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        const wsUrl = `${protocol}//${host}/ws/${id}`;
        console.log(`Connecting to WebSocket: ${wsUrl}`);
        this.websocket = new WebSocket(wsUrl);
        this.websocketConnected = false;
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected successfully');
            this.websocketConnected = true;
        };
        
        this.websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log('WebSocket message received:', data);
                this.handleInterviewUpdate(data);
            } catch (error) {
                console.error('WebSocket message error:', error, event.data);
            }
        };
        
        this.websocket.onclose = (event) => {
            console.log('WebSocket disconnected', event.code, event.reason);
            this.websocketConnected = false;
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.websocketConnected = false;
            this.showStatus('WebSocket connection failed', 'error');
        };
    }

    waitForWebSocketConnection(timeout = 5000) {
        return new Promise((resolve, reject) => {
            const checkInterval = 100;
            let elapsed = 0;
            
            const checkConnection = () => {
                if (this.websocketConnected) {
                    console.log('WebSocket connection confirmed');
                    resolve();
                } else if (elapsed >= timeout) {
                    reject(new Error('WebSocket connection timeout'));
                } else {
                    elapsed += checkInterval;
                    setTimeout(checkConnection, checkInterval);
                }
            };
            
            checkConnection();
        });
    }

    transitionToMainApp() {
        this.isStarted = true;
        this.elements.startScreen.style.display = 'none';
        this.elements.appContainer.style.display = 'flex';
        
        // Trigger fade-in animation
        setTimeout(() => {
            this.elements.appContainer.style.opacity = '1';
        }, 50);
    }

    showInitialGreeting() {
        // Backend will handle the interview flow via WebSocket
        this.elements.questionText.textContent = 'Connecting to interview system...';
        // Hide timer initially
        this.elements.timeBarContainer.style.display = 'none';
        
        // Add timeout in case WebSocket messages don't arrive
        setTimeout(() => {
            if (this.elements.questionText.textContent === 'Connecting to interview system...') {
                this.elements.questionText.textContent = 'Connection timeout. Please check the server logs or try again.';
                this.showStatus('WebSocket connection or interview start timeout', 'error');
            }
        }, 30000); // 30 second timeout
    }

    handleInterviewUpdate(data) {
        console.log('WebSocket message received:', data.type, data.data);
        
        switch (data.type) {
            case 'status_update':
                this.handleStatusUpdate(data.data);
                break;
            case 'interview_started':
                this.handleInterviewStarted(data.data);
                break;
            case 'stage_started':
                this.handleStageStarted(data.data);
                break;
            case 'question_delivered':
                this.handleQuestionDelivered(data.data);
                break;
            case 'timer_started':
                this.handleTimerStarted(data.data);
                break;
            case 'timer_update':
                this.handleTimerUpdate(data.data);
                break;
            case 'response_collected':
                this.handleResponseCollected(data.data);
                break;
            case 'interview_completed':
                this.handleInterviewComplete();
                break;
            case 'interview_error':
                this.handleInterviewError(data.data);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }

    handleStatusUpdate(status) {
        if (status.current_stage) {
            this.updateStage(this.formatStageName(status.current_stage));
        }
        if (status.current_question) {
            this.elements.questionText.textContent = status.current_question;
        }
    }

    handleInterviewStarted(data) {
        console.log('Interview started:', data);
        this.showStatus('Interview started successfully!', 'success');
        setTimeout(() => this.hideStatus(), 3000);
    }

    handleStageStarted(data) {
        console.log('Stage started:', data);
        this.currentStage = data.stage;
        this.elements.questionText.textContent = data.question;
        this.updateStage(data.stage_name);
        this.totalTime = data.time_limit;
        
        // Clear any existing timer
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        
        // Show timer container if time limit > 0
        if (data.time_limit > 0) {
            this.elements.timeBarContainer.style.display = 'block';
            console.log('Timer bar container made visible');
        } else {
            this.elements.timeBarContainer.style.display = 'none';
            console.log('Timer bar container hidden (no time limit)');
        }
    }

    handleQuestionDelivered(data) {
        console.log('Question delivered:', data);
        // Question is already displayed from stage_started
    }

    handleTimerStarted(data) {
        console.log('Timer started:', data);
        this.timeRemaining = data.time_remaining;
        this.totalTime = data.time_limit;
        this.updateTimerDisplay();
    }

    handleTimerUpdate(data) {
        console.log('Timer update received:', data);
        this.timeRemaining = data.time_remaining;
        this.updateTimerDisplay();
        console.log(`Timer updated: ${Math.ceil(this.timeRemaining)}s remaining`);
    }

    handleResponseCollected(data) {
        console.log('Response collected:', data);
        // Timer will naturally end, no need to clear
    }

    handleInterviewError(data) {
        console.log('Interview error:', data);
        this.showStatus(`Interview error: ${data.message}`, 'error');
    }

    updateStage(stageName) {
        this.elements.currentStage.textContent = stageName;
        this.elements.stageIndicator.style.display = 'block';
    }

    formatStageName(stage) {
        return stage.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    updateTimerDisplay() {
        const seconds = Math.ceil(this.timeRemaining);
        this.elements.timeLabel.textContent = `${seconds}s left`;
        
        // Update progress bar
        const progress = (this.timeRemaining / this.totalTime) * 100;
        this.elements.timeBar.style.width = `${progress}%`;
        
        console.log(`Timer display updated: ${seconds}s left, ${progress.toFixed(1)}% width, totalTime: ${this.totalTime}`);
        
        // Update progress bar color based on time remaining
        this.elements.timeBar.className = 'time-bar';
        if (progress <= 25) {
            this.elements.timeBar.classList.add('danger');
        } else if (progress <= 50) {
            this.elements.timeBar.classList.add('warning');
        }
    }


    handleInterviewComplete() {
        this.cleanup();
        this.elements.questionText.textContent = 'Interview completed! Thank you for your time.';
        this.updateStage('Completed');
        this.elements.timeLabel.textContent = 'Done';
        this.elements.timeBar.style.width = '100%';
        this.showStatus('Interview completed successfully!', 'success');
    }

    showStatus(message, type = '') {
        this.elements.statusMessage.textContent = message;
        this.elements.statusSection.className = `status-section ${type}`;
        this.elements.statusSection.style.display = 'block';
        
        // Auto-hide after 5 seconds for non-error messages
        if (type !== 'error') {
            setTimeout(() => this.hideStatus(), 5000);
        }
    }

    hideStatus() {
        this.elements.statusSection.style.display = 'none';
    }

    cleanup() {
        // Clear timer
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        
        // Close WebSocket
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        // Stop video stream
        if (this.videoStream) {
            this.videoStream.getTracks().forEach(track => track.stop());
            this.videoStream = null;
        }
        
        // Reset state
        this.interviewId = null;
        this.currentStage = null;
        this.timeRemaining = 0;
        this.totalTime = 0;
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.autoInterviewApp = new AutoInterviewApp();
});
