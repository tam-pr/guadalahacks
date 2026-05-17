import { useEffect, useRef, useState } from "react";

export default function CameraStream() {
  const socketRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [prediction, setPrediction] = useState("Waiting...");
  const [confidence, setConfidence] = useState(0);
  const [sentence, setSentence] = useState("");
  const [currentMode, setCurrentMode] = useState("WORDS");
  const [videoFrame, setVideoFrame] = useState(null);
  
  // 🎙️ NEW AUDIO STATES
  const [audioEnabled, setAudioEnabled] = useState(true);
  const prevSentenceRef = useRef("");

  // ==========================================
  // WEBSOCKET CONNECTION
  // ==========================================
  useEffect(() => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws");
    socketRef.current = ws;

    ws.onopen = () => setConnected(true);
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.prediction) setPrediction(data.prediction);
        if (data.confidence !== undefined) setConfidence(data.confidence);
        if (data.sentence !== undefined) setSentence(data.sentence);
        if (data.mode) setCurrentMode(data.mode);
        if (data.image) setVideoFrame(`data:image/jpeg;base64,${data.image}`);
      } catch (err) {
        console.error("Error reading backend message:", err);
      }
    };

    ws.onclose = () => setConnected(false);
    return () => ws.close();
  }, []);

  // ==========================================
  // 🎙️ TEXT-TO-SPEECH ENGINE
  // ==========================================
  useEffect(() => {
    if (!audioEnabled) {
      prevSentenceRef.current = sentence;
      return;
    }

    // Check if new text was ADDED to the sentence
    if (sentence.length > prevSentenceRef.current.length) {
      // Slice out ONLY the brand new word or letter
      const addedText = sentence.slice(prevSentenceRef.current.length).trim();
      
      if (addedText) {
        window.speechSynthesis.cancel(); // Stop current speech to avoid overlapping echoes
        const utterance = new SpeechSynthesisUtterance(addedText);
        utterance.lang = "es-MX"; // Mexican Spanish Native Pronunciation
        utterance.rate = 1.0; 
        window.speechSynthesis.speak(utterance);
      }
    } 
    // If the sentence was erased completely (Global Fist Erase)
    else if (sentence.length === 0 && prevSentenceRef.current.length > 0) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance("Borrado");
        utterance.lang = "es-MX";
        window.speechSynthesis.speak(utterance);
    }
    
    prevSentenceRef.current = sentence;
  }, [sentence, audioEnabled]);

  const sendCommand = (cmd) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(cmd);
    }
  };

  const readFullSentence = () => {
    if (sentence.trim() !== "") {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(sentence);
      utterance.lang = "es-MX";
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "20px", fontFamily: "sans-serif" }}>
      
      {/* STATUS & AUDIO BANNERS */}
      <div style={{ display: "flex", justifyContent: "center", gap: "15px", marginBottom: "15px" }}>
        <span style={{ padding: "8px 16px", borderRadius: "20px", fontWeight: "bold", backgroundColor: connected ? "#10B981" : "#EF4444", color: "white" }}>
          {connected ? "AI PIPELINE: ACTIVE" : "AI PIPELINE: OFFLINE"}
        </span>
        <button 
          onClick={() => setAudioEnabled(!audioEnabled)}
          style={{ padding: "8px 16px", borderRadius: "20px", fontWeight: "bold", cursor: "pointer", border: "none", backgroundColor: audioEnabled ? "#8B5CF6" : "#6B7280", color: "white" }}
        >
          {audioEnabled ? "🔊 Voice: ON" : "🔇 Voice: MUTED"}
        </button>
      </div>

      {/* THE CAMERA MONITOR */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: "20px" }}>
        {videoFrame ? (
          <img src={videoFrame} alt="AI Camera Feed" style={{ width: "450px", borderRadius: "12px", boxShadow: "0 4px 12px rgba(0,0,0,0.2)" }} />
        ) : (
          <div style={{ width: "450px", height: "337px", backgroundColor: "#222", borderRadius: "12px", display: "flex", alignItems: "center", justifyContent: "center", color: "#666" }}>
            Waiting for Python Camera Feed...
          </div>
        )}
      </div>

      {/* CONTROL ACTIONS */}
      <div style={{ display: "flex", gap: "10px", justifyContent: "center" }}>
        <button 
          onClick={() => sendCommand("TOGGLE_MODE")} 
          style={{ padding: "12px 24px", fontWeight: "bold", cursor: "pointer", backgroundColor: "#0070f3", color: "white", borderRadius: "6px", border: "none" }}
        >
          Toggle Mode (Active: {currentMode})
        </button>
        
        <button 
          onClick={() => sendCommand("CLEAR_SENTENCE")} 
          style={{ padding: "12px 24px", fontWeight: "bold", cursor: "pointer", backgroundColor: "#dc2626", color: "white", borderRadius: "6px", border: "none" }}
        >
          Clear Sentence
        </button>
      </div>

      {/* AI PREDICTION OUTPUT */}
      <div style={{ marginTop: "30px" }}>
        <h3>Current Sign: <span style={{ color: "#0070f3", textTransform: "capitalize" }}>{prediction.replace(/_/g, ' ')}</span></h3>
        <h4 style={{ color: confidence >= 75 ? "#10B981" : "#F59E0B" }}>Confidence: {confidence.toFixed(1)}%</h4>
        
        <div style={{ marginTop: "20px", padding: "25px", backgroundColor: "#f8fafc", borderRadius: "8px", border: "2px solid #e2e8f0" }}>
          <h3 style={{ margin: "0 0 10px 0", color: "#475569" }}>Assembled Sentence:</h3>
          <p style={{ fontSize: "28px", fontWeight: "bold", color: "#0f172a", minHeight: "40px", margin: 0 }}>
            {sentence || "Waiting for signs..."}
          </p>
          
          <button 
            onClick={readFullSentence}
            style={{ marginTop: "15px", padding: "8px 16px", cursor: "pointer", backgroundColor: "#e2e8f0", color: "#334155", border: "1px solid #cbd5e1", borderRadius: "4px", fontWeight: "bold" }}
          >
            ▶️ Play Full Sentence
          </button>
        </div>
      </div>
    </div>
  );
}