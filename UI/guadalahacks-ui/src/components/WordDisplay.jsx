import { useEffect, useState } from "react";
import { getSocket } from "./socket"; 

export default function WordDisplay() {
  const [sentence, setSentence] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const socket = getSocket();

    socket.onopen = () => {
      console.log("Connected to AI Backend!");
      setLoading(false);
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.sentence !== undefined) {
          setSentence(data.sentence);
        }
      } catch (error) {
        console.error("Error parsing AI data:", error);
      }
    };

    return () => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.close();
      }
    };
  }, []);

  if (loading) {
    return <h3>Connecting to AI Engine...</h3>;
  }

  return (
    <div style={{ padding: "20px", textAlign: "center", backgroundColor: "#f5f5f5", borderRadius: "12px", border: "2px solid black", marginTop: "20px" }}>
      <h2>Live Translation</h2>
      <h1 style={{ color: "#0070f3", fontSize: "32px" }}>{sentence || "Waiting for signs..."}</h1>
    </div>
  );
}