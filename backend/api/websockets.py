from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict
import json
import asyncio

class ConnectionManager:
    def __init__(self):
        # session_id -> list of active websockets
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)

    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

    async def broadcast(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            # We use a copy of the list to avoid issues if a connection drops during broadcast
            disconnected = []
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            
            for connection in disconnected:
                self.disconnect(connection, session_id)

manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        # Heartbeat task
        async def keep_alive():
            try:
                while True:
                    await asyncio.sleep(30)
                    await websocket.send_json({"type": "PING"})
            except Exception:
                pass

        heartbeat = asyncio.create_task(keep_alive())

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # When a panelist updates a score, broadcast it to others in the same session
            if message.get("type") == "SCORE_UPDATE":
                # Ensure we include a timestamp for race condition handling
                if "timestamp" not in message:
                    import time
                    message["timestamp"] = time.time()
                await manager.broadcast(message, session_id)
            elif message.get("type") == "PONG":
                pass # Just keep-alive
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, session_id)
    finally:
        if 'heartbeat' in locals():
            heartbeat.cancel()
