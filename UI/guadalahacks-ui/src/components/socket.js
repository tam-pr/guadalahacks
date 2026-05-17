let socket;

export function getSocket() {
  if (!socket || socket.readyState === WebSocket.CLOSED) {
    socket = new WebSocket("ws://127.0.0.1:8000/ws");
  }
  return socket;
}