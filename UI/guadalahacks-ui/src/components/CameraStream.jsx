import { useEffect, useRef, useState } from "react";

export default function CameraStream() {
  const videoRef = useRef(null);

  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState("");

  // front/back phone camera
  const [facingMode, setFacingMode] = useState("environment");

  // AI mode
  const [mode, setMode] = useState("spelling");

  // translated output
  const [translatedText, setTranslatedText] = useState("Waiting for translation...");

  // detect mobile
  const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const startCamera = async () => {
    try {
      setStreaming(true);
      setError("");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode,
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;

        // Safari/iPhone fix
        await videoRef.current.play();
      }

      // fake AI output test
      setTimeout(() => {
        if (mode === "spelling") {
          setTranslatedText("H-E-L-L-O");
        } else {
          setTranslatedText("Hello");
        }
      }, 2000);

    } catch (err) {
      console.error(err);
      setError("Could not access camera");
      setStreaming(false);
    }
  };

  const stopCamera = () => {
    if (!videoRef.current) return;

    const stream = videoRef.current.srcObject;

    if (stream instanceof MediaStream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    videoRef.current.srcObject = null;
    setStreaming(false);
  };

  const switchCamera = async () => {
    const newFacingMode = facingMode === "environment" ? "user" : "environment";
    setFacingMode(newFacingMode);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: newFacingMode,
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setStreaming(true);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={styles.container}>
      <h1>Sign Language Translator</h1>

      {error && <p style={styles.error}>{error}</p>}

      {/* Force mirroring on both cameras so your right hand always appears on your right side */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          ...styles.video,
          display: streaming ? "block" : "none",
          transform: "scaleX(-1)" 
        }}
      />

      {/* CAMERA BUTTONS */}
      <div style={styles.buttons}>
        <button onClick={startCamera}>Open Camera</button>
        <button onClick={stopCamera}>Stop Camera</button>
        {isMobile && <button onClick={switchCamera}>Switch Camera</button>}
      </div>

      {/* MODE SWITCH */}
      <div style={styles.buttons}>
        <button
          onClick={() => setMode("spelling")}
          style={mode === "spelling" ? styles.activeButton : styles.button}
        >
          Spelling
        </button>

        <button
          onClick={() => setMode("phrases")}
          style={mode === "phrases" ? styles.activeButton : styles.button}
        >
          Phrases
        </button>
      </div>

      <p>
        Current Mode: <strong>{mode}</strong>
      </p>

      <p>Status: {streaming ? "Streaming" : "Stopped"}</p>

      {/* TRANSLATION OUTPUT */}
      <div style={styles.translationBox}>
        <h2>Translation</h2>
        <p>{translatedText}</p>
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "16px",
    padding: "20px",
    fontFamily: "Arial",
  },

  video: {
    width: "100%",
    maxWidth: "700px",
    borderRadius: "12px",
    border: "2px solid black",
    backgroundColor: "black",
  },

  buttons: {
    display: "flex",
    gap: "10px",
    flexWrap: "wrap",
  },

  button: {
    padding: "10px 18px",
    border: "1px solid black",
    background: "white",
    cursor: "pointer",
    borderRadius: "8px",
  },

  activeButton: {
    padding: "10px 18px",
    border: "1px solid black",
    background: "black",
    color: "white",
    cursor: "pointer",
    borderRadius: "8px",
  },

  translationBox: {
    marginTop: "20px",
    padding: "20px",
    width: "100%",
    maxWidth: "700px",
    border: "2px solid black",
    borderRadius: "12px",
    backgroundColor: "#f5f5f5",
  },

  error: {
    color: "red",
  },
};
