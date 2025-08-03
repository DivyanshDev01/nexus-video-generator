import { useState, useRef } from 'react';

const VideoGenerator = () => {
  const [prompt, setPrompt] = useState('');
  const [videoUrl, setVideoUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const videoRef = useRef(null);

  const generateVideo = async () => {
    setLoading(true);
    setError(null);
    setProgress(0);
    setStatus('Starting generation...');
    
    try {
      // Start the generation process
      const response = await fetch(`${import.meta.env.VITE_API_URL}/create_project`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenes: [{ 
            prompt, 
            duration: 5.0, 
            resolution: "720p",
            sampling_steps: 25
          }],
          transitions: [{ 
            type: "dissolve", 
            duration: 1.0 
          }]
        }),
      });

      const data = await response.json();
      const taskId = data.task_id;
      setStatus('Processing your video...');

      // Poll for task status
      const checkStatus = async () => {
        try {
          const statusResponse = await fetch(
            `${import.meta.env.VITE_API_URL}/task_status/${taskId}`
          );
          const statusData = await statusResponse.json();
          
          setProgress(statusData.progress);
          setStatus(statusData.status);
          
          if (statusData.status === 'completed') {
            setVideoUrl(statusData.video_url);
            setLoading(false);
          } else if (statusData.status.startsWith('failed')) {
            setError('Video generation failed');
            setLoading(false);
          } else {
            setTimeout(checkStatus, 3000);
          }
        } catch (err) {
          setError('Failed to check status');
          setLoading(false);
        }
      };

      checkStatus();
    } catch (err) {
      setError('Failed to start generation');
      setLoading(false);
    }
  };

  return (
    <div className="generator-container">
      <div className="input-group">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe your scene (e.g., cyberpunk city at night)"
          disabled={loading}
        />
        <button 
          onClick={generateVideo} 
          disabled={loading || !prompt.trim()}
          className={loading ? 'loading' : ''}
        >
          {loading ? (
            <>
              <span className="spinner"></span>
              Generating...
            </>
          ) : 'Generate Video'}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="progress-container">
        {loading && (
          <>
            <div className="progress-bar" style={{ width: `${progress}%` }}></div>
            <div className="status-message">{status}</div>
          </>
        )}
      </div>

      {videoUrl && (
        <div className="video-container">
          <video 
            ref={videoRef}
            controls 
            src={videoUrl} 
            onPlay={() => videoRef.current?.play()}
          >
            Your browser does not support the video tag.
          </video>
          <a 
            href={videoUrl} 
            download="nexus-video.mp4"
            className="download-btn"
          >
            Download Video
          </a>
        </div>
      )}
    </div>
  );
};

export default VideoGenerator;
