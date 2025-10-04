#!/usr/bin/env python3
"""
Real-time Miku Bot Dashboard with FastAPI WebSockets
Instant updates, no polling, direct bot-to-dashboard communication
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import datetime
from typing import List, Dict, Any
import uvicorn
import logging
import pickle
from pathlib import Path
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Miku Bot Dashboard", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="templates"), name="static")

# Global state
connected_clients: List[WebSocket] = []
bot_status: Dict[str, Any] = {
    "online": False,
    "guilds": [],
    "users": 0,
    "uptime": None,
    "latency": None,
    "voice_channels": [],
    "in_voice": False,
    "last_seen": None,
    "timestamp": None
}

class ConnectionManager:
    """Manages WebSocket connections and broadcasting"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
        
        # Send current status immediately
        await self.send_personal_message(json.dumps(bot_status), websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to a specific WebSocket"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected WebSockets"""
        if not self.active_connections:
            return
            
        # Create a copy of the list to avoid modification during iteration
        connections_to_remove = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                connections_to_remove.append(connection)
        
        # Remove failed connections
        for connection in connections_to_remove:
            self.disconnect(connection)

# Global connection manager
manager = ConnectionManager()

@app.get("/")
async def get_dashboard():
    """Serve the modern dashboard HTML"""
    logger.info("Serving modern dashboard HTML...")
    try:
        with open("templates/dashboard.html", "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"Successfully read modern dashboard template, length: {len(content)}")
        
        # Add cache-busting headers to prevent browser caching
        response = HTMLResponse(content=content)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    except Exception as e:
        logger.error(f"Error reading modern dashboard template: {e}")
        return HTMLResponse(content="<h1>Error loading dashboard</h1><p>Check server logs for details.</p>")

@app.get("/enhanced")
async def get_enhanced_dashboard():
    """Serve the enhanced dashboard with Penguin UI components"""
    logger.info("Serving enhanced dashboard HTML...")
    try:
        with open("templates/dashboard_enhanced.html", "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"Successfully read enhanced dashboard template, length: {len(content)}")
        
        # Add cache-busting headers to prevent browser caching
        response = HTMLResponse(content=content)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    except Exception as e:
        logger.error(f"Error reading enhanced dashboard template: {e}")
        return HTMLResponse(content="<h1>Error loading enhanced dashboard</h1><p>Check server logs for details.</p>")

@app.get("/test")
async def get_test():
    """Test endpoint"""
    return HTMLResponse(content=open("templates/test.html", "r", encoding="utf-8").read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            try:
                data = await websocket.receive_text()
                logger.info(f"Received WebSocket message: {data}")
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        manager.disconnect(websocket)

@app.post("/api/bot/status")
async def update_bot_status(status_data: Dict[str, Any]):
    """Endpoint for bot to send status updates"""
    global bot_status
    
    # Update global status
    bot_status.update(status_data)
    bot_status["timestamp"] = datetime.datetime.now().isoformat()
    bot_status["last_seen"] = datetime.datetime.now().isoformat()
    
    logger.info(f"Bot status updated: {status_data}")
    
    # Broadcast to all connected clients
    await manager.broadcast(json.dumps(bot_status))
    
    return {"success": True, "message": "Status updated and broadcasted"}

@app.get("/api/status")
async def get_current_status():
    """Get current bot status"""
    return bot_status

@app.get("/api/stream")
async def stream_status_legacy():
    """Legacy endpoint for old dashboard"""
    return {"message": "Use WebSocket /ws for real-time updates"}

@app.get("/api/logs/stream")
async def stream_logs_legacy():
    """Legacy endpoint for old dashboard"""
    return {"message": "Use WebSocket /ws/logs for real-time log streaming"}

@app.get("/api/refresh")
async def refresh_status_legacy():
    """Legacy endpoint for old dashboard"""
    return {"success": True, "status": bot_status}

@app.get("/api/logs")
async def get_logs():
    """Get all bot logs as JSON"""
    try:
        with open("discord_bot.log", "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split('\n')
            return {
                "success": True,
                "logs": lines,
                "total_lines": len(lines)
            }
    except FileNotFoundError:
        return {"success": False, "error": "Log file not found"}
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/bot/voice-channels")
async def get_voice_channels():
    """Get available voice channels"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:5001/voice-channels') as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Bot API returned status {response.status}"}
    except Exception as e:
        logger.error(f"Error getting voice channels: {e}")
        return {"error": str(e)}

@app.post("/api/bot/join-voice")
async def join_voice_channel(request: Dict[str, Any]):
    """Join a specific voice channel"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post('http://localhost:5001/join-voice', 
                                 json=request) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", f"Bot API returned status {response.status}")}
    except Exception as e:
        logger.error(f"Error joining voice channel: {e}")
        return {"error": str(e)}

@app.post("/api/bot/leave-voice")
async def leave_voice_channel():
    """Leave current voice channel"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post('http://localhost:5001/leave-voice') as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", f"Bot API returned status {response.status}")}
    except Exception as e:
        logger.error(f"Error leaving voice channel: {e}")
        return {"error": str(e)}

@app.post("/api/bot/deafen")
async def deafen_bot():
    """Deafen the bot (stop it from hearing voice)"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post('http://localhost:5001/deafen') as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", f"Bot API returned status {response.status}")}
    except Exception as e:
        logger.error(f"Error deafening bot: {e}")
        return {"error": str(e)}

@app.post("/api/bot/undeafen")
async def undeafen_bot():
    """Undeafen the bot (allow it to hear voice again)"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post('http://localhost:5001/undeafen') as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", f"Bot API returned status {response.status}")}
    except Exception as e:
        logger.error(f"Error undeafening bot: {e}")
        return {"error": str(e)}

@app.post("/api/bot/interrupt")
async def interrupt_bot():
    """Interrupt the bot's current speaking/processing"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post('http://localhost:5001/interrupt', 
                                 timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"Bot API error: {response.status} - {error_text}")
                    return {"error": f"Bot API error: {response.status}"}
    except Exception as e:
        logger.error(f"Error interrupting bot: {e}")
        return {"error": str(e)}

@app.post("/api/bot/event")
async def bot_event(event_data: Dict[str, Any]):
    """Endpoint for bot to send specific events"""
    event_type = event_data.get("type")
    
    if event_type == "voice_join":
        # Bot joined a voice channel
        voice_info = event_data.get("voice_info", {})
        bot_status["voice_channels"].append(voice_info)
        bot_status["in_voice"] = True
        
    elif event_type == "voice_leave":
        # Bot left voice channel
        bot_status["voice_channels"] = []
        bot_status["in_voice"] = False
        
    elif event_type == "disconnect":
        # Bot disconnected
        bot_status["online"] = False
        bot_status["voice_channels"] = []
        bot_status["in_voice"] = False
        
    elif event_type == "connect":
        # Bot connected
        bot_status["online"] = True
        
    elif event_type == "guild_update":
        # Guild information updated
        bot_status["guilds"] = event_data.get("guilds", [])
        bot_status["users"] = event_data.get("users", 0)
    
    # Update timestamp
    bot_status["timestamp"] = datetime.datetime.now().isoformat()
    bot_status["last_seen"] = datetime.datetime.now().isoformat()
    
    logger.info(f"Bot event received: {event_type}")
    
    # Broadcast to all connected clients
    await manager.broadcast(json.dumps(bot_status))
    
    return {"success": True, "message": f"Event {event_type} processed"}

@app.post("/api/bot/conversation")
async def bot_conversation(conversation_data: Dict[str, Any]):
    """Receive conversation updates from bot"""
    try:
        # Broadcast conversation update to all connected clients
        message = {
            "type": "conversation",
            "data": conversation_data
        }
        await manager.broadcast(json.dumps(message))
        logger.info(f"Broadcasted conversation update: {conversation_data.get('type', 'unknown')}")
        return {"success": True}
    except Exception as e:
        logger.error(f"Error broadcasting conversation: {e}")
        return {"error": str(e)}

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming"""
    await websocket.accept()
    try:
        # Send existing logs first
        with open("discord_bot.log", "r", encoding="utf-8") as f:
            lines = f.readlines()
            recent_lines = lines[-20:] if len(lines) > 20 else lines
            for line in recent_lines:
                await websocket.send_text(json.dumps({"log": line.strip()}))
        
        # Monitor for new lines
        with open("discord_bot.log", "r", encoding="utf-8") as f:
            f.seek(0, 2)  # Go to end of file
            while True:
                line = f.readline()
                if line:
                    await websocket.send_text(json.dumps({"log": line.strip()}))
                else:
                    await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Error in log streaming: {e}")

# Memory Management Endpoints
@app.get("/memory")
async def get_memory_dashboard():
    """Serve the memory management dashboard"""
    try:
        with open("templates/memory_dashboard.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        logger.error(f"Error loading memory dashboard: {e}")
        return HTMLResponse(content="<h1>Error loading memory dashboard</h1>", status_code=500)

@app.get("/api/memory/conversations")
async def get_memory_conversations():
    """Get all conversations from memory"""
    try:
        memory_path = Path("waifu_memory")
        if not memory_path.exists():
            return {"success": False, "error": "Memory directory not found"}
        
        with open(memory_path / "memory_store.pkl", "rb") as f:
            memory_store = pickle.load(f)
        
        conversations = []
        if 'conversations' in memory_store:
            for conv in memory_store['conversations']:
                conversation_data = {
                    'conversation_id': conv.get('conversation_id'),
                    'user_id': conv.get('user_id'),
                    'username': conv.get('username', f"User_{conv.get('user_id', 'Unknown')}"),
                    'guild_id': conv.get('guild_id'),
                    'timestamp': conv.get('timestamp').isoformat() if hasattr(conv.get('timestamp'), 'isoformat') else str(conv.get('timestamp')),
                    'user_message': conv.get('user_message'),
                    'assistant_message': conv.get('assistant_message')
                }
                conversations.append(conversation_data)
        
        return {
            "success": True,
            "conversations": conversations,
            "total": len(conversations)
        }
    except Exception as e:
        logger.error(f"Error loading conversations: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/memory/search")
async def search_memories(query: str, limit: int = 10):
    """Search memories by text content"""
    try:
        memory_path = Path("waifu_memory")
        if not memory_path.exists():
            return {"success": False, "error": "Memory directory not found"}
        
        with open(memory_path / "memory_store.pkl", "rb") as f:
            memory_store = pickle.load(f)
        
        results = []
        if 'conversations' in memory_store:
            query_lower = query.lower()
            for conv in memory_store['conversations']:
                user_msg = conv.get('user_message', '').lower()
                bot_msg = conv.get('assistant_message', '').lower()
                
                if query_lower in user_msg or query_lower in bot_msg:
                    conversation_data = {
                        'conversation_id': conv.get('conversation_id'),
                        'user_id': conv.get('user_id'),
                        'username': conv.get('username', f"User_{conv.get('user_id', 'Unknown')}"),
                        'guild_id': conv.get('guild_id'),
                        'timestamp': conv.get('timestamp').isoformat() if hasattr(conv.get('timestamp'), 'isoformat') else str(conv.get('timestamp')),
                        'user_message': conv.get('user_message'),
                        'assistant_message': conv.get('assistant_message')
                    }
                    results.append(conversation_data)
                    
                    if len(results) >= limit:
                        break
        
        return {
            "success": True,
            "results": results,
            "total": len(results),
            "query": query
        }
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/memory/stats")
async def get_memory_stats():
    """Get memory statistics"""
    try:
        memory_path = Path("waifu_memory")
        if not memory_path.exists():
            return {"success": False, "error": "Memory directory not found"}
        
        with open(memory_path / "memory_store.pkl", "rb") as f:
            memory_store = pickle.load(f)
        
        conversations = memory_store.get('conversations', [])
        memories = memory_store.get('memories', [])
        
        # Calculate statistics
        unique_users = set(conv.get('user_id') for conv in conversations)
        unique_guilds = set(conv.get('guild_id') for conv in conversations)
        
        # User counts
        user_counts = {}
        for conv in conversations:
            user_id = conv.get('user_id')
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
        
        # Guild counts
        guild_counts = {}
        for conv in conversations:
            guild_id = conv.get('guild_id')
            guild_counts[guild_id] = guild_counts.get(guild_id, 0) + 1
        
        return {
            "success": True,
            "stats": {
                "total_conversations": len(conversations),
                "total_memories": len(memories),
                "unique_users": len(unique_users),
                "unique_guilds": len(unique_guilds),
                "user_counts": user_counts,
                "guild_counts": guild_counts
            }
        }
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/memory/user-mapping")
async def get_user_mapping():
    """Get user ID to name mapping"""
    try:
        # This would ideally come from Discord API or stored user data
        # For now, return a static mapping based on known users
        user_mapping = {
            "140238123324932096": {
                "name": "Primary User",
                "display_name": "Primary User",
                "avatar": None
            },
            "bennyhunnid": {
                "name": "Benny",
                "display_name": "Benny",
                "avatar": None
            }
        }
        
        guild_mapping = {
            "0": {
                "name": "Direct Message",
                "display_name": "Direct Message"
            },
            "1239394551756755055": {
                "name": "Beautiful City Server",
                "display_name": "Beautiful City Server"
            }
        }
        
        return {
            "success": True,
            "users": user_mapping,
            "guilds": guild_mapping
        }
    except Exception as e:
        logger.error(f"Error getting user mapping: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    logger.info("🚀 Starting Miku Bot Dashboard (FastAPI + WebSockets)")
    logger.info("📊 Dashboard will be available at: http://localhost:5002")
    logger.info("🔌 WebSocket endpoint: ws://localhost:5002/ws")
    logger.info("📡 Bot API endpoint: http://localhost:5002/api/bot/status")
    
    uvicorn.run(
        "dashboard_fastapi:app",
        host="0.0.0.0",
        port=5002,
        reload=False,
        log_level="info"
    )
