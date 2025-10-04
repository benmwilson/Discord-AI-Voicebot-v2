# 🚀 Miku Bot Dashboard - FastAPI WebSockets Edition

## Real-time Dashboard with Instant Updates

This is a completely reimagined dashboard system using **FastAPI + WebSockets** for instant, real-time updates. No more polling, no more delays - everything updates instantly when the bot status changes.

## 🎯 Key Features

- **⚡ Instant Updates** - WebSocket-based real-time communication
- **🔄 No Polling** - Bot pushes updates directly to dashboard
- **📊 Real-time Status** - Voice, guild, uptime updates instantly
- **📋 Live Logs** - Real-time log streaming
- **🎮 Interactive Controls** - Direct bot communication
- **🔌 Auto-reconnect** - Handles connection drops gracefully

## 🏗️ Architecture

```
Bot (Discord.py) → HTTP POST → FastAPI Server → WebSocket → Dashboard (Browser)
     ↓                    ↓                        ↓
  Status Events    Real-time Processing    Instant UI Updates
```

### Flow:
1. **Bot detects event** (voice join/leave, status change, etc.)
2. **Bot sends HTTP POST** to FastAPI server with status data
3. **FastAPI server broadcasts** to all connected WebSocket clients
4. **Dashboard updates instantly** in the browser

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install -r dashboard_fastapi_requirements.txt
```

### 2. Start the Bot
```bash
python Main.py
```

### 3. Start the Dashboard
```bash
python start_dashboard_fastapi.py
```

### 4. Access Dashboard
Open your browser: `http://localhost:5002`

## 📡 API Endpoints

### Bot → Dashboard Communication

#### `POST /api/bot/status`
Bot sends complete status update:
```json
{
  "online": true,
  "guilds": [...],
  "users": 123,
  "uptime": "1:23:45",
  "latency": 125.5,
  "voice_channels": [...],
  "in_voice": true,
  "timestamp": "2025-10-03T18:30:00"
}
```

#### `POST /api/bot/event`
Bot sends specific events:
```json
{
  "type": "voice_join",
  "voice_info": {
    "channel_name": "General",
    "guild_name": "My Server"
  }
}
```

### Dashboard → Bot Communication

#### `GET /api/status`
Get current bot status

#### `WebSocket /ws`
Real-time status updates

#### `WebSocket /ws/logs`
Real-time log streaming

## 🔧 Bot Integration

The bot automatically sends updates when:

- **Bot connects/disconnects** → `connect`/`disconnect` events
- **Voice channel changes** → `voice_join`/`voice_leave` events  
- **Status updates** → Complete status data
- **Guild changes** → `guild_update` events

### Example Bot Code:
```python
# Bot automatically sends updates via:
await self.send_dashboard_update()        # Full status
await self.send_dashboard_event('voice_join', voice_info)  # Specific event
```

## 🎮 Dashboard Features

### Real-time Status Display
- **Bot Status** - Online/Offline with real-time updates
- **Voice Status** - Shows current voice channels instantly
- **Guild Information** - Server count and member count
- **Uptime** - Live uptime counter
- **Latency** - Discord API latency

### Interactive Controls
- **Join/Leave Voice** - Direct voice channel control
- **Status Refresh** - Manual status update request
- **Connection Test** - Test WebSocket connection
- **Live Logs** - Real-time log streaming

### Visual Indicators
- **Connection Status** - Shows WebSocket connection state
- **Real-time Badges** - Status indicators update instantly
- **Activity Timestamps** - When events occurred
- **Auto-reconnect** - Handles connection drops

## 🔌 WebSocket Events

### Client → Server
- **Connection** - Client connects to `/ws`
- **Ping** - Keep-alive messages

### Server → Client
- **Status Update** - Complete bot status
- **Event Notification** - Specific bot events
- **Log Entry** - New log lines (via `/ws/logs`)

## 🛠️ Development

### File Structure
```
dashboard_fastapi.py          # FastAPI server with WebSockets
templates/dashboard_fastapi.html  # Dashboard HTML with WebSocket client
start_dashboard_fastapi.py    # Startup script
dashboard_fastapi_requirements.txt  # Dependencies
```

### Key Components

#### FastAPI Server (`dashboard_fastapi.py`)
- WebSocket connection management
- Real-time broadcasting
- Bot API endpoints
- Log streaming

#### Dashboard Client (`templates/dashboard_fastapi.html`)
- WebSocket client
- Real-time UI updates
- Auto-reconnection
- Interactive controls

#### Bot Integration (`Main.py`)
- HTTP POST to dashboard
- Event-driven updates
- Status broadcasting

## 🚨 Troubleshooting

### Connection Issues
```bash
# Check if dashboard is running
curl http://localhost:5002/api/status

# Check WebSocket connection
# Open browser console and look for WebSocket messages
```

### Bot Not Sending Updates
```bash
# Check bot logs for dashboard communication errors
# Look for "Dashboard update sent successfully" messages
```

### WebSocket Disconnection
- Dashboard automatically reconnects
- Check browser console for reconnection attempts
- Use "Test Connection" button to verify

## 🎯 Benefits Over Old System

| Feature | Old System | New System |
|---------|------------|------------|
| Updates | 2-5 second delay | Instant |
| Method | Polling | Push-based |
| Efficiency | High CPU usage | Low CPU usage |
| Reliability | Cache issues | Real-time |
| Scalability | Limited | Unlimited clients |

## 🔮 Future Enhancements

- **Redis Pub/Sub** - For multiple bot instances
- **Authentication** - Secure dashboard access
- **Metrics** - Performance monitoring
- **Alerts** - Status notifications
- **Mobile App** - Native mobile dashboard

---

## 🎉 Enjoy Your Real-time Dashboard!

No more polling, no more delays - just instant, real-time updates! 🚀
