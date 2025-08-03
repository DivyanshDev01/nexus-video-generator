import { useState } from 'react'
import VideoGenerator from './components/VideoGenerator'
import './App.css'

function App() {
  return (
    <div className="app-container">
      <header>
        <h1>NEXUS Video Generator</h1>
        <p>Create AI-generated videos from text prompts</p>
      </header>
      <main>
        <VideoGenerator />
      </main>
      <footer>
        <p>© 2025 NEXUS Creative Studio</p>
      </footer>
    </div>
  )
}

export default App
