import discord
import json
import ollama
import datetime
import subprocess
import time
import aiohttp
import base64
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import pickle
import asyncio
import numpy as np
from discord.ext import commands, voice_recv
from faster_whisper import WhisperModel
import wave
from sentence_transformers import SentenceTransformer
import faiss
from typing import Optional, Dict, Any, List
import logging
from aiohttp import web
import threading

# Discord Bot Token
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')  # Load from environment variable

# API Server Configuration
API_HOST = "localhost"
API_PORT = 5001

# System Prompt - Load from external file
def load_system_prompt():
    try:
        # Try to load custom system prompt first
        with open('system_prompt.txt', 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback to example system prompt
        with open('system_prompt.example.txt', 'r', encoding='utf-8') as f:
            print("📝 Using example system prompt (copy system_prompt.example.txt to system_prompt.txt to customize)")
            return f.read().strip()

SYSTEM_PROMPT = load_system_prompt()

class EnhancedMemoryStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        import torch
        
        # Skip GPU for now due to PyTorch CUDA issues, use fast CPU model
        device = 'cpu'
        print("💻 Using CPU for embeddings (stable mode)")
        
        # Use smaller, faster model for better performance
        fast_model = "all-MiniLM-L6-v2"  # Small and fast
        self.encoder = SentenceTransformer(fast_model, device=device)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # For small datasets, CPU FAISS is often faster than GPU overhead
        # RTX 4060 is better used for the embedding model
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        print(f"📊 Using CPU FAISS (optimal for small datasets), GPU for embeddings")
        
        # Increased threshold for similarity matching
        self.similarity_threshold = 0.90  # Higher = fewer matches = faster
        
        # Maximum number of memories to return  
        self.max_memories = 2  # Reduced for speed
        
        # Initialize other components
        self.base_path = Path("waifu_memory")
        self.base_path.mkdir(exist_ok=True)
        self.conversations = []
        self.memories = []
        self.conversation_index = {}
        self.logger = logging.getLogger('MemoryStore')
        
        self.load_memories()

    def get_conversation_context(self, user_id: str, current_message: str, 
                               guild_id: Optional[int] = None,
                               max_context: int = 3) -> str:
        try:
            # Get relevant memories with stricter filtering
            relevant = self.search_memories(current_message, k=max_context * 2)
            
            # Filter memories more strictly
            user_memories = []
            for memory in relevant:
                mem = memory["memory"]
                relevance = memory["relevance"]
                
                # Only include highly relevant memories
                if (mem["user_id"] == user_id and 
                    relevance > self.similarity_threshold and
                    (guild_id is None or mem["guild_id"] == guild_id)):
                    user_memories.append(memory)
            
            # Limit number of memories
            user_memories = user_memories[:max_context]
            
            # Sort by both relevance and recency
            user_memories.sort(key=lambda x: (
                x["relevance"],
                x["memory"]["timestamp"]
            ), reverse=True)
            
            # Build context with relevance scores
            if user_memories:
                context = "Previous relevant conversations:\n\n"
                for memory in user_memories:
                    relevance = memory["relevance"]
                    timestamp = memory["memory"]["timestamp"]
                    if isinstance(timestamp, str):
                        timestamp = datetime.datetime.fromisoformat(timestamp)
                    time_ago = datetime.datetime.now() - timestamp
                    
                    # Only include if relevance is high enough
                    if relevance > self.similarity_threshold:
                        context += f"{memory['memory']['text']}\n---\n"
            else:
                context = ""
                
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting conversation context: {e}")
            return ""

    def search_memories(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.embed_text(query)
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(self.memories):
                    memory = self.memories[idx]
                    
                    # Convert distance to similarity score (0-1)
                    similarity = 1 / (1 + distance)
                    
                    # Only include if similarity is above threshold
                    if similarity > self.similarity_threshold:
                        results.append({
                            "memory": memory,
                            "relevance": similarity
                        })
            
            return sorted(results, key=lambda x: x["relevance"], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error searching memories: {e}")
            return []

    def add_conversation_turn(self, user_id: str, timestamp: datetime, 
                            user_message: str, assistant_message: str,
                            guild_id: Optional[int] = None, username: Optional[str] = None):
        try:
            # Check for similar existing memories first
            existing_memories = self.search_memories(user_message + " " + assistant_message)
            
            # Only add if this is sufficiently different from existing memories
            if not any(mem["relevance"] > self.similarity_threshold for mem in existing_memories):
                conversation = {
                    "user_id": user_id,
                    "username": username or f"User_{user_id}",
                    "guild_id": guild_id,
                    "timestamp": timestamp,
                    "user_message": user_message,
                    "assistant_message": assistant_message,
                    "conversation_id": len(self.conversations)
                }
                
                self.conversations.append(conversation)
                
                combined_text = f"User: {user_message}\nAssistant: {assistant_message}"
                embedding = self.embed_text(combined_text)
                
                memory_id = len(self.memories)
                self.memories.append({
                    "id": memory_id,
                    "text": combined_text,
                    "conversation_id": conversation["conversation_id"],
                    "timestamp": timestamp,
                    "user_id": user_id,
                    "username": username or f"User_{user_id}",
                    "guild_id": guild_id
                })
                
                self.index.add(embedding.reshape(1, -1))
                self.conversation_index[conversation["conversation_id"]] = memory_id
                
                self.save_memories()
                return True
            else:
                self.logger.info("Similar memory already exists, skipping addition")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding conversation turn: {e}")
            return False

    def embed_text(self, text: str) -> np.ndarray:
        return self.encoder.encode(text)

    def save_memories(self):
        try:
            save_data = {
                "memories": self.memories,
                "conversations": self.conversations,
                "conversation_index": self.conversation_index
            }
            
            memory_path = self.base_path / "memory_store.pkl"
            index_path = self.base_path / "faiss_index.pkl"
            backup_path = self.base_path / "memory_store.backup.pkl"
            
            if memory_path.exists():
                memory_path.rename(backup_path)
            
            with open(memory_path, 'wb') as f:
                pickle.dump(save_data, f)
                
            faiss.write_index(self.index, str(index_path))
            
            if backup_path.exists():
                backup_path.unlink()
                
        except Exception as e:
            self.logger.error(f"Error saving memories: {e}")
            if backup_path.exists():
                backup_path.rename(memory_path)

    def load_memories(self):
        try:
            memory_path = self.base_path / "memory_store.pkl"
            index_path = self.base_path / "faiss_index.pkl"
            
            if memory_path.exists() and index_path.exists():
                with open(memory_path, 'rb') as f:
                    save_data = pickle.load(f)
                    
                self.memories = save_data["memories"]
                self.conversations = save_data["conversations"]
                self.conversation_index = save_data["conversation_index"]
                self.index = faiss.read_index(str(index_path))
                
        except Exception as e:
            self.logger.error(f"Error loading memories: {e}")
            self.memories = []
            self.conversations = []
            self.conversation_index = {}
            self.index = faiss.IndexFlatL2(self.embedding_dim)

    def clear_memories(self, guild_id: Optional[int] = None, user_id: Optional[str] = None):
        try:
            if guild_id is None and user_id is None:
                self.memories = []
                self.conversations = []
                self.conversation_index = {}
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            else:
                new_memories = []
                new_index_data = []
                for memory in self.memories:
                    if ((guild_id is None or memory["guild_id"] != guild_id) and 
                        (user_id is None or memory["user_id"] != user_id)):
                        new_memories.append(memory)
                        embedding = self.embed_text(memory["text"])
                        new_index_data.append(embedding)
                
                self.memories = new_memories
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                if new_index_data:
                    self.index.add(np.vstack(new_index_data))
                
            self.save_memories()
            
        except Exception as e:
            self.logger.error(f"Error clearing memories: {e}")

class UnifiedConversationHandler:
    def __init__(self, memory_store: EnhancedMemoryStore):
        self.memory_store = memory_store
        self.logger = logging.getLogger('UnifiedConversationHandler')

    async def process_interaction(
        self,
        user_id: str,
        guild_id: int,
        message_content: str,
        interaction_type: str = "chat",
        context: Optional[dict] = None,
        username: Optional[str] = None
    ):
        try:
            # Get historical context from memory store
            conversation_context = self.memory_store.get_conversation_context(
                user_id=user_id,
                current_message=message_content,
                guild_id=guild_id
            )

            # Combine memory context with recent messages
            recent_context = ""
            if context and 'recent_messages' in context:
                recent_context = "\n".join([
                    f"{msg['author']}: {msg['content']}"
                    for msg in context['recent_messages'][-5:]
                ])

            # Prepare the complete context
            full_context = f"{SYSTEM_PROMPT}\n\nMemory context:\n{conversation_context}\nRecent conversation:\n{recent_context}"

            # Always generate response for direct mentions, replies, or pings
            messages = [
                {'role': 'system', 'content': full_context},
                {'role': 'user', 'content': f"{user_id}: {message_content}"}
            ]
            response = await self._generate_response(messages)

            if response:
                # Store the interaction in memory
                self.memory_store.add_conversation_turn(
                    user_id=user_id,
                    timestamp=datetime.datetime.now(),
                    user_message=message_content,
                    assistant_message=response,
                    guild_id=guild_id,
                    username=username
                )

            return {
                'should_respond': True,
                'response': response
            }

        except Exception as e:
            self.logger.error(f"Error processing interaction: {e}")
            return {
                'should_respond': False,
                'response': None
            }

    async def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.chat(
                    model='hf.co/subsectmusic/qwriko3-4b-instruct-2507:Q4_K_M',
                    messages=messages,
                    options={
                        'num_predict': 2048,        # Maximum number of tokens to generate
                        'temperature': 0.8,         # Higher temperature (0-1) increases creativity/randomness
                        'top_k': 40,               # Limit vocabulary to top K options per token
                        'top_p': 0.9,              # Nucleus sampling threshold
                        'repeat_penalty': 1.1,     # Penalize repetition (>1.0 reduces repetition)
                        'presence_penalty': 0.2,    # Penalize tokens based on presence in context
                        'frequency_penalty': 0.2    # Penalize tokens based on frequency in context
                    }
                )
            )
            return response['message']['content']
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return None


class ChannelContext:
    def __init__(self, max_messages=10):
        self.messages = []
        self.max_messages = max_messages
        self.last_bot_message = None
        self.logger = logging.getLogger('ChannelContext')
    
    def add_message(self, message, is_bot=False):
        message_data = {
            'author': str(message.author),
            'content': message.content,
            'timestamp': message.created_at.isoformat(),
            'is_bot': is_bot
        }
        
        if is_bot:
            self.last_bot_message = message_data
            
        self.messages.append(message_data)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
    
    def get_context(self):
        return {
            'recent_messages': self.messages[-10:],
            'last_bot_message': self.last_bot_message
        }

    def was_last_message_from_bot(self):
        return self.messages and self.messages[-1].get('is_bot', False)

class BotAPIServer:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.app = web.Application()
        self.setup_routes()
        
    def setup_routes(self):
        """Setup API routes"""
        self.app.router.add_get('/status', self.get_status)
        self.app.router.add_post('/command', self.handle_command)
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/voice-channels', self.get_voice_channels)
        self.app.router.add_post('/join-voice', self.join_voice_channel)
        self.app.router.add_post('/leave-voice', self.leave_voice_channel)
        self.app.router.add_post('/deafen', self.deafen_bot)
        self.app.router.add_post('/undeafen', self.undeafen_bot)
        self.app.router.add_post('/interrupt', self.interrupt_bot)
        
    async def get_status(self, request):
        """Get bot status"""
        try:
            guilds = []
            total_users = 0
            voice_channels = []
            
            # Check if bot is online using our custom status
            is_online = getattr(self.bot, 'is_online', False) and self.bot.is_ready()
            
            if is_online:
                for guild in self.bot.guilds:
                    guild_info = {
                        'id': guild.id,
                        'name': guild.name,
                        'member_count': guild.member_count,
                        'icon_url': str(guild.icon.url) if guild.icon else None
                    }
                    guilds.append(guild_info)
                    total_users += guild.member_count
                
            # Check voice channel connections - use both bot.voice_clients and cog tracking
            for voice_client in self.bot.voice_clients:
                if voice_client and voice_client.channel and voice_client.is_connected():
                    voice_info = {
                        'guild_id': voice_client.guild.id,
                        'guild_name': voice_client.guild.name,
                        'channel_id': voice_client.channel.id,
                        'channel_name': voice_client.channel.name,
                        'connected': voice_client.is_connected(),
                        'playing': voice_client.is_playing()
                    }
                    voice_channels.append(voice_info)
            
            # Also check the voice cog's active voice clients for more accurate tracking
            voice_cog = self.bot.get_cog('Testing')
            if voice_cog and hasattr(voice_cog, 'active_voice_clients'):
                for guild_id, vc in voice_cog.active_voice_clients.items():
                    if vc and vc.is_connected() and vc.channel:
                        # Check if this voice client is already in our list
                        already_tracked = any(
                            vc_info['guild_id'] == vc.guild.id and vc_info['channel_id'] == vc.channel.id
                            for vc_info in voice_channels
                        )
                        if not already_tracked:
                            voice_info = {
                                'guild_id': vc.guild.id,
                                'guild_name': vc.guild.name,
                                'channel_id': vc.channel.id,
                                'channel_name': vc.channel.name,
                                'connected': vc.is_connected(),
                                'playing': vc.is_playing()
                            }
                            voice_channels.append(voice_info)
                
                uptime = datetime.datetime.now() - self.bot.start_time if hasattr(self.bot, 'start_time') else None
                uptime_str = str(uptime).split('.')[0] if uptime else None
                
                # More accurate voice detection
                in_voice = len(voice_channels) > 0 and any(vc.get('connected', False) for vc in voice_channels)
                
                status_data = {
                    'online': True,
                    'guilds': guilds,
                    'users': total_users,
                    'uptime': uptime_str,
                    'latency': round(self.bot.latency * 1000, 2) if self.bot.latency else None,
                    'voice_channels': voice_channels,
                    'in_voice': in_voice,
                    'timestamp': datetime.datetime.now().isoformat()
                }
            else:
                status_data = {
                    'online': False,
                    'guilds': [],
                    'users': 0,
                    'uptime': None,
                    'latency': None,
                    'voice_channels': [],
                    'in_voice': False,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
            return web.json_response(status_data)
            
        except Exception as e:
            return web.json_response({
                'error': str(e),
                'online': False
            }, status=500)
    
    async def handle_command(self, request):
        """Handle bot commands"""
        try:
            data = await request.json()
            command = data.get('command')
            
            if command == 'restart':
                # Note: This would require implementing a restart mechanism
                return web.json_response({
                    'success': True,
                    'message': 'Restart command received (not implemented yet)'
                })
            elif command == 'status':
                return await self.get_status(request)
            else:
                return web.json_response({
                    'success': False,
                    'error': f'Unknown command: {command}'
                })
                
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def health_check(self, request):
        """Health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'bot_ready': self.bot.is_ready(),
            'timestamp': datetime.datetime.now().isoformat()
        })

    async def get_voice_channels(self, request):
        """Get available voice channels from all guilds"""
        try:
            channels = []
            for guild in self.bot.guilds:
                for channel in guild.voice_channels:
                    channels.append({
                        'id': str(channel.id),
                        'name': channel.name,
                        'guild_id': str(guild.id),
                        'guild_name': guild.name,
                        'user_count': len(channel.members)
                    })
            return web.json_response(channels)
        except Exception as e:
            self.bot.logger.error(f"Error getting voice channels: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def join_voice_channel(self, request):
        """Join a specific voice channel"""
        try:
            self.bot.logger.info("=== JOIN VOICE REQUEST RECEIVED ===")
            data = await request.json()
            self.bot.logger.info(f"Request data: {data}")
            channel_id = int(data.get('channel_id'))
            self.bot.logger.info(f"Channel ID: {channel_id}")
            
            # Find the channel
            channel = None
            for guild in self.bot.guilds:
                for vc in guild.voice_channels:
                    if vc.id == channel_id:
                        channel = vc
                        break
                if channel:
                    break
            
            if not channel:
                return web.json_response({"error": "Voice channel not found"}, status=404)
            
            # Get the voice cog (it's actually named 'Testing')
            voice_cog = self.bot.get_cog('Testing')
            if not voice_cog:
                return web.json_response({"error": "Voice cog not available"}, status=500)
            
            # Directly implement the voice joining logic (copied from !vc command)
            self.bot.logger.info(f"Joining channel: {channel}")
            
            await voice_cog.piper.initialize()
            self.bot.logger.info("Piper initialized")
            
            # Disconnect any existing connection first
            if channel.guild.voice_client:
                await channel.guild.voice_client.disconnect()
                await asyncio.sleep(1)
            
            # Connect with new discord.py version (includes 4006 fix)
            vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
            self.bot.logger.info("Connected to voice channel")
            
            # Track voice client for multi-user management
            guild_id = channel.guild.id
            voice_cog.active_voice_clients[guild_id] = vc
            
            # Wait for connection to stabilize
            await asyncio.sleep(2)
            
            # Verify connection before proceeding
            if not vc.is_connected():
                raise Exception("Voice connection failed to establish properly")
            
            # Create voice sink and start listening
            sink = VoiceSink(voice_cog, self.bot)
            self.bot.logger.info("Created multi-user voice sink")
            
            vc.listen(sink)
            self.bot.logger.info("Started listening for multiple users")
            
            # Wait a moment for voice connection to stabilize
            await asyncio.sleep(0.5)
            
            # Store sink reference for later control (after listen is called)
            vc.sink = sink
            
            # Send immediate dashboard update
            self.bot.logger.info("Sending immediate dashboard update after voice connection")
            await self.bot.send_dashboard_update()
            
            return web.json_response({
                "success": True,
                "channel_name": channel.name,
                "guild_name": channel.guild.name
            })
            
        except Exception as e:
            import traceback
            self.bot.logger.error(f"Error joining voice channel: {e}")
            self.bot.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Clean up any partial connection
            try:
                if 'channel' in locals() and channel.guild.voice_client:
                    await channel.guild.voice_client.disconnect()
            except:
                pass
                
            return web.json_response({"error": str(e)}, status=500)

    async def leave_voice_channel(self, request):
        """Leave current voice channel"""
        try:
            # Find any active voice client
            voice_client = None
            for vc in self.bot.voice_clients:
                if vc.is_connected():
                    voice_client = vc
                    break
            
            if not voice_client:
                return web.json_response({"error": "Not connected to any voice channel"}, status=400)
            
            # Get the voice cog for cleanup (it's actually named 'Testing')
            voice_cog = self.bot.get_cog('Testing')
            guild_id = voice_client.guild.id
            
            # Clean up voice client tracking (same as !stop command)
            if voice_cog and guild_id in voice_cog.active_voice_clients:
                del voice_cog.active_voice_clients[guild_id]
            
            # Clear user sessions for this guild
            if voice_cog:
                users_to_remove = [user_id for user_id, session in voice_cog.user_sessions.items() 
                                 if session.get('voice_channel') and session['voice_channel'].guild.id == guild_id]
                for user_id in users_to_remove:
                    del voice_cog.user_sessions[user_id]
            
            # Disconnect the voice client
            await voice_client.disconnect()
            
            # Send immediate dashboard update
            self.bot.logger.info("Sending immediate dashboard update after voice disconnection")
            await self.bot.send_dashboard_update()
            
            return web.json_response({
                "success": True,
                "message": "Left voice channel"
            })
            
        except Exception as e:
            self.bot.logger.error(f"Error leaving voice channel: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def deafen_bot(self, request):
        """Deafen the bot (stop it from hearing voice)"""
        try:
            # Get the voice cog
            voice_cog = self.bot.get_cog('Testing')
            if not voice_cog:
                return web.json_response({"error": "Voice cog not available"}, status=500)
            
            # Check if bot is in voice
            if not voice_cog.active_voice_clients:
                return web.json_response({"error": "Bot is not in a voice channel"}, status=400)
            
            # Deafen all active voice clients
            deafened_count = 0
            for guild_id, vc in voice_cog.active_voice_clients.items():
                if vc and vc.is_connected():
                    await vc.guild.change_voice_state(channel=vc.channel, self_deaf=True)
                    
                    # Disable VoiceSink to stop listening
                    if hasattr(vc, 'sink') and vc.sink:
                        vc.sink.decode = False
                        self.bot.logger.info(f"VoiceSink disabled in guild {guild_id}")
                    else:
                        self.bot.logger.warning(f"VoiceSink not found in guild {guild_id} - deafen may not stop listening")
                    
                    deafened_count += 1
                    self.bot.logger.info(f"Deafened bot in guild {guild_id}")
            
            if deafened_count > 0:
                # Send immediate dashboard update
                await self.bot.send_dashboard_update()
                return web.json_response({
                    "success": True,
                    "message": f"Bot deafened in {deafened_count} voice channel(s)"
                })
            else:
                return web.json_response({"error": "No active voice connections to deafen"}, status=400)
                
        except Exception as e:
            self.bot.logger.error(f"Error deafening bot: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def undeafen_bot(self, request):
        """Undeafen the bot (allow it to hear voice again)"""
        try:
            # Get the voice cog
            voice_cog = self.bot.get_cog('Testing')
            if not voice_cog:
                return web.json_response({"error": "Voice cog not available"}, status=500)
            
            # Check if bot is in voice
            if not voice_cog.active_voice_clients:
                return web.json_response({"error": "Bot is not in a voice channel"}, status=400)
            
            # Undeafen all active voice clients
            undeafened_count = 0
            for guild_id, vc in voice_cog.active_voice_clients.items():
                if vc and vc.is_connected():
                    await vc.guild.change_voice_state(channel=vc.channel, self_deaf=False)
                    
                    # Re-enable VoiceSink to resume listening
                    if hasattr(vc, 'sink') and vc.sink:
                        vc.sink.decode = True
                        self.bot.logger.info(f"VoiceSink enabled in guild {guild_id}")
                    else:
                        self.bot.logger.warning(f"VoiceSink not found in guild {guild_id} - undeafen may not resume listening")
                    
                    undeafened_count += 1
                    self.bot.logger.info(f"Undeafened bot in guild {guild_id}")
            
            if undeafened_count > 0:
                # Send immediate dashboard update
                await self.bot.send_dashboard_update()
                return web.json_response({
                    "success": True,
                    "message": f"Bot undeafened in {undeafened_count} voice channel(s)"
                })
            else:
                return web.json_response({"error": "No active voice connections to undeafen"}, status=400)
                
        except Exception as e:
            self.bot.logger.error(f"Error undeafening bot: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def interrupt_bot(self, request):
        """Interrupt the bot's current speaking/processing"""
        try:
            # Get the voice cog
            voice_cog = self.bot.get_cog('Testing')
            if not voice_cog:
                return web.json_response({"error": "Voice cog not available"}, status=500)
            
            # Check if bot is in voice
            if not voice_cog.active_voice_clients:
                return web.json_response({"error": "Bot is not in a voice channel"}, status=400)
            
            self.bot.logger.info("Interrupt request received from dashboard")
            
            # Interrupt the bot
            success = await voice_cog.interrupt_bot()
            
            if success:
                self.bot.logger.info("Bot interrupted successfully via API")
                return web.json_response({
                    "success": True,
                    "message": "Bot interrupted successfully"
                })
            else:
                self.bot.logger.warning("No active processing to interrupt")
                return web.json_response({"error": "No active processing to interrupt"}, status=400)
                
        except Exception as e:
            self.bot.logger.error(f"Error interrupting bot: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def start_server(self):
        """Start the API server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, API_HOST, API_PORT)
        await site.start()
        print(f"🌐 Bot API server started on http://{API_HOST}:{API_PORT}")
        return runner

class Bot(commands.Bot):
    def __init__(self, command_prefix, intents, memory_store=None):
        super().__init__(command_prefix=commands.when_mentioned_or('!'), intents=intents)
        self.channel_contexts = {}
        self.piper = PiperTTS()
        self.audio_processor = AudioProcessor()
        self.api_server = None
        self.start_time = None
        self.is_online = False  # Initialize as offline
        # Use pre-loaded memory store if provided, otherwise create new one
        self.memory_store = memory_store if memory_store else EnhancedMemoryStore()
        self.conversation_handler = UnifiedConversationHandler(self.memory_store)
        self.logger = logging.getLogger('Bot')
        self.is_processing = False
    def scrub_bot_username(self, text: str) -> str:
        """
        Removes the bot's username and mentions from the message.
        """
        if not hasattr(self, 'user'):
            return text  # Return the original text if the bot's user is not initialized yet

        # Remove bot mentions (e.g., <@123456789012345678>)
        text = text.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "")
        
        # Remove bot username (e.g., Hikari-Chan#1660)
        bot_username = f"{self.user.name}#{self.user.discriminator}"
        text = text.replace(bot_username, "")
        
        # Clean up any extra spaces
        text = " ".join(text.split())
        return text        

    async def on_ready(self):
        self.start_time = datetime.datetime.now()
        self.logger.info('Logged in as {0.id}/{0}'.format(self.user))
        self.logger.info('Commands:')
        self.logger.info('- !vc - Join voice and start listening')
        self.logger.info('- !stop - Disconnect from voice')
        self.logger.info('- !deafen - Deafen bot (stop hearing voice)')
        self.logger.info('- !undeafen - Undeafen bot (allow hearing voice)')
        self.logger.info('- !die  - Shutdown bot')
        self.logger.info('------')
        
        # Start API server
        self.api_server = BotAPIServer(self)
        await self.api_server.start_server()
        
        # Set online status
        self.is_online = True
        self.logger.info('Bot status: ONLINE')
        
        # Send real-time update to dashboard
        await self.send_dashboard_update()
        
        # Start periodic status updates (every 10 seconds)
        self.loop.create_task(self.periodic_status_update())

    async def on_disconnect(self):
        """Called when the bot disconnects from Discord"""
        self.is_online = False
        self.logger.warning('Bot status: OFFLINE - Disconnected from Discord')
        await self.send_dashboard_event('disconnect')
        
    async def on_resumed(self):
        """Called when the bot reconnects to Discord"""
        self.is_online = True
        self.logger.info('Bot status: ONLINE - Reconnected to Discord')
        await self.send_dashboard_event('connect')
        await self.send_dashboard_update()
        
    async def on_error(self, event, *args, **kwargs):
        """Called when an error occurs"""
        self.logger.error(f'Bot error in {event}: {args}')
        
    async def on_command_error(self, ctx, error):
        """Called when a command error occurs"""
        self.logger.error(f'Command error: {error}')

    async def on_voice_state_update(self, member, before, after):
        """Handle voice state changes"""
        # Only track our own voice state changes
        if member == self.user:
            if before.channel != after.channel:
                if after.channel:
                    # Bot joined a voice channel
                    await self.send_dashboard_update()
                    self.logger.info(f"Bot joined voice channel: {after.channel.name}")
                    
                    # Send session start event
                    await self.send_conversation_update({
                        'type': 'session_start',
                        'guild_id': after.channel.guild.id,
                        'guild_name': after.channel.guild.name,
                        'channel_id': after.channel.id,
                        'channel_name': after.channel.name,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                else:
                    # Bot left voice channel
                    await self.send_dashboard_update()
                    self.logger.info(f"Bot left voice channel: {before.channel.name}")
                    
                    # Send session end event
                    await self.send_conversation_update({
                        'type': 'session_end',
                        'guild_id': before.channel.guild.id,
                        'guild_name': before.channel.guild.name,
                        'channel_id': before.channel.id,
                        'channel_name': before.channel.name,
                        'timestamp': datetime.datetime.now().isoformat()
                    })

    async def periodic_status_update(self):
        """Send periodic status updates to dashboard"""
        while True:
            try:
                await asyncio.sleep(2)  # Update every 2 seconds for more responsive updates
                if self.is_online:
                    await self.send_dashboard_update()
            except Exception as e:
                self.logger.error(f"Error in periodic status update: {e}")
    
    async def send_conversation_update(self, data):
        """Send conversation update to dashboard"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post('http://localhost:8001/api/conversation', json=data) as response:
                    if response.status == 200:
                        self.logger.info("Conversation update sent successfully")
                    else:
                        self.logger.warning(f"Failed to send conversation update: {response.status}")
        except Exception as e:
            self.logger.error(f"Error sending conversation update: {e}")

    async def send_dashboard_update(self):
        """Send real-time update to dashboard"""
        try:
            import aiohttp
            
            # Prepare status data
            guilds = []
            total_users = 0
            voice_channels = []
            
            for guild in self.guilds:
                guild_info = {
                    'id': guild.id,
                    'name': guild.name,
                    'member_count': guild.member_count,
                    'icon_url': str(guild.icon.url) if guild.icon else None
                }
                guilds.append(guild_info)
                total_users += guild.member_count
            
            # Get voice channel information - use both bot.voice_clients and cog tracking
            for voice_client in self.voice_clients:
                if voice_client and voice_client.channel and voice_client.is_connected():
                    voice_info = {
                        'guild_id': voice_client.guild.id,
                        'guild_name': voice_client.guild.name,
                        'channel_id': voice_client.channel.id,
                        'channel_name': voice_client.channel.name,
                        'connected': voice_client.is_connected(),
                        'playing': voice_client.is_playing()
                    }
                    voice_channels.append(voice_info)
            
            # Also check the voice cog's active voice clients for more accurate tracking
            voice_cog = self.get_cog('Testing')
            if voice_cog and hasattr(voice_cog, 'active_voice_clients'):
                for guild_id, vc in voice_cog.active_voice_clients.items():
                    if vc and vc.is_connected() and vc.channel:
                        # Check if this voice client is already in our list
                        already_tracked = any(
                            vc_info['guild_id'] == vc.guild.id and vc_info['channel_id'] == vc.channel.id
                            for vc_info in voice_channels
                        )
                        if not already_tracked:
                            voice_info = {
                                'guild_id': vc.guild.id,
                                'guild_name': vc.guild.name,
                                'channel_id': vc.channel.id,
                                'channel_name': vc.channel.name,
                                'connected': vc.is_connected(),
                                'playing': vc.is_playing()
                            }
                            voice_channels.append(voice_info)
            
            # Calculate uptime
            uptime = None
            if hasattr(self, 'start_time') and self.start_time:
                uptime_delta = datetime.datetime.now() - self.start_time
                hours, remainder = divmod(int(uptime_delta.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                uptime = f"{hours}:{minutes:02d}:{seconds:02d}"
            
            # Determine if bot is actually in voice
            in_voice = len(voice_channels) > 0 and any(vc.get('connected', False) for vc in voice_channels)
            
            # Check deafen status and processing state from voice clients
            is_deafened = False
            processing_state = "idle"
            current_activity = ""
            processing_queue = []
            queue_size = 0
            
            if in_voice and voice_cog and hasattr(voice_cog, 'active_voice_clients'):
                for guild_id, vc in voice_cog.active_voice_clients.items():
                    if vc and vc.is_connected():
                        # Check if bot is deafened (self_deaf property)
                        if hasattr(vc, 'self_deaf') and vc.self_deaf:
                            is_deafened = True
                        # Also check VoiceSink decode status as backup
                        elif hasattr(vc, 'sink') and vc.sink and not vc.sink.decode:
                            is_deafened = True
                        
                        # Get processing state for this guild
                        if hasattr(voice_cog, 'get_processing_state'):
                            processing_state = voice_cog.get_processing_state(guild_id)
                        if hasattr(voice_cog, 'get_current_activity'):
                            current_activity = voice_cog.get_current_activity(guild_id)
                        if hasattr(voice_cog, 'get_processing_queue'):
                            processing_queue = voice_cog.get_processing_queue(guild_id)
                        if hasattr(voice_cog, 'get_queue_size'):
                            queue_size = voice_cog.get_queue_size(guild_id)
                        break
            
            status_data = {
                'online': self.is_online,
                'guilds': guilds,
                'users': total_users,
                'uptime': uptime,
                'latency': round(self.latency * 1000, 2) if self.latency else None,
                'voice_channels': voice_channels,
                'in_voice': in_voice,
                'is_deafened': is_deafened,
                'processing_state': processing_state,
                'current_activity': current_activity,
                'processing_queue': processing_queue,
                'queue_size': queue_size,
                'timestamp': datetime.datetime.now().isoformat(),
                'update_type': 'immediate'  # Mark immediate updates
            }
            
            # Send to dashboard
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post('http://localhost:5002/api/bot/status', 
                                          json=status_data, 
                                          timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            self.logger.info('Dashboard update sent successfully')
                        else:
                            self.logger.warning(f'Dashboard update failed: {response.status}')
                except Exception as e:
                    self.logger.warning(f'Failed to send dashboard update: {e}')
                    
        except Exception as e:
            self.logger.error(f'Error sending dashboard update: {e}')

    async def send_conversation_update(self, conversation_data):
        """Send conversation update to dashboard"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post('http://localhost:5002/api/bot/conversation', 
                                     json=conversation_data, 
                                     timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        self.logger.info("Conversation update sent successfully")
                    else:
                        self.logger.warning(f"Conversation update failed: {response.status}")
        except Exception as e:
            self.logger.error(f"Error sending conversation update: {e}")
    
    async def send_dashboard_event(self, event_type: str, event_data: dict = None):
        """Send specific event to dashboard"""
        try:
            import aiohttp
            
            event_payload = {
                'type': event_type,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            if event_data:
                event_payload.update(event_data)
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post('http://localhost:5002/api/bot/event', 
                                          json=event_payload, 
                                          timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            self.logger.info(f'Dashboard event sent: {event_type}')
                        else:
                            self.logger.warning(f'Dashboard event failed: {response.status}')
                except Exception as e:
                    self.logger.warning(f'Failed to send dashboard event: {e}')
                    
        except Exception as e:
            self.logger.error(f'Error sending dashboard event: {e}')

    async def process_message(self, message, response_content, use_tts=True):
        self.is_processing = True
        
        try:
            result = await self.conversation_handler.process_interaction(
                user_id=str(message.author),
                guild_id=message.guild.id,
                message_content=message.content,
                interaction_type="direct_mention" if use_tts else "chat",
                context=self.channel_contexts[message.channel.id].get_context()
            )
            
            if result['should_respond'] and result['response']:
                response_content = result['response']
                response_content = response_content.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "")
                
                async with message.channel.typing():
                    if use_tts:
                        wav_file = await self.piper.generate_speech(response_content)
                        if wav_file:
                            await self.send_voice_message(message.channel, wav_file, response_content)
                        else:
                            await message.channel.send(response_content)
                    else:
                        await message.channel.send(response_content)
                    
                    self.channel_contexts[message.channel.id].add_message(
                        message=type('obj', (object,), {
                            'author': self.user,
                            'content': response_content,
                            'created_at': datetime.datetime.now()
                        }),
                        is_bot=True
                    )
        finally:
            self.is_processing = False

    async def send_voice_message(self, channel, wav_file, response_text):
        try:
            ogg_file = wav_file.replace('.wav', '.ogg')
            waveform, duration = self.audio_processor.fast_convert_and_analyze(wav_file, ogg_file)

            file_size = os.path.getsize(ogg_file)

            async with aiohttp.ClientSession() as session:
                upload_url_endpoint = f'https://discord.com/api/v10/channels/{channel.id}/attachments'
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bot {self.http.token}'
                }
                data = {
                    "files": [{
                        "filename": "voice-message.ogg",
                        "file_size": file_size,
                        "id": "2"
                    }]
                }
                
                async with session.post(upload_url_endpoint, headers=headers, json=data) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to get upload URL: {await resp.text()}")
                    upload_data = await resp.json()
                    
                    upload_url = upload_data['attachments'][0]['upload_url']
                    upload_filename = upload_data['attachments'][0]['upload_filename']

                headers = {
                    'Content-Type': 'audio/ogg',
                    'Authorization': f'Bot {self.http.token}'
                }
                with open(ogg_file, 'rb') as f:
                    async with session.put(upload_url, headers=headers, data=f.read()) as resp:
                        if resp.status != 200:
                            raise Exception(f"Failed to upload file: {await resp.text()}")

                data = {
                    "flags": 8192,
                    "attachments": [{
                        "id": "0",
                        "filename": "voice-message.ogg",
                        "uploaded_filename": upload_filename,
                        "duration_secs": duration,
                        "waveform": waveform
                    }]
                }

                message_endpoint = f'https://discord.com/api/v10/channels/{channel.id}/messages'
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bot {self.http.token}'
                }
                async with session.post(message_endpoint, headers=headers, json=data) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to send voice message: {await resp.text()}")

                await channel.send(content=response_text)

        except Exception as e:
            self.logger.error(f"Voice message error: {e}")
            await channel.send(content=response_text)
            await self.play_voice_fallback(channel, wav_file, response_text)
        finally:
            try:
                os.remove(wav_file)
                os.remove(ogg_file)
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")

    async def play_voice_fallback(self, channel, wav_file, response_text):
        try:
            voice_channel = None
            for vc in channel.guild.voice_channels:
                if len(vc.members) > 0:
                    voice_channel = vc
                    break

            if not voice_channel:
                await channel.send("No available voice channel to join.")
                return

            voice_client = await voice_channel.connect()
            source = discord.FFmpegPCMAudio(wav_file)
            voice_client.play(source, after=lambda e: print(f'Player error: {e}') if e else None)

            while voice_client.is_playing():
                await asyncio.sleep(1)

            await voice_client.disconnect()

        except Exception as e:
            self.logger.error(f"Voice fallback error: {e}")
            await channel.send("Failed to play voice message in a voice channel.")

class Testing(commands.Cog):
    def __init__(self, bot, piper):
        self.bot = bot
        self.piper = piper
        self.conversation_handler = UnifiedConversationHandler(bot.memory_store)
        self.logger = logging.getLogger('VoiceCog')
        
        # Multi-user session management
        self.user_sessions = {}  # user_id -> session data
        self.active_voice_clients = {}  # guild_id -> voice_client
        
        # LLM protection queue (protect that RTX 4060!)
        self.llm_semaphore = asyncio.Semaphore(1)  # Only 1 LLM call at a time
        self.llm_queue_size = 0
        
        # Processing state tracking for dashboard
        self.processing_states = {}  # guild_id -> current processing state
        self.current_activity = {}  # guild_id -> current activity description
        self.speaking_states = {}  # guild_id -> whether bot is currently speaking
        self.concurrent_activities = {}  # guild_id -> list of concurrent activities
        self.processing_queue = {}  # guild_id -> list of queued activities
        self.queue_size = {}  # guild_id -> number of items in queue
        
        print("Multi-user Voice cog initialized with LLM protection")

    def get_user_session(self, user_id):
        """Get or create user session data"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'conversation_history': [],
                'last_interaction': time.time(),
                'voice_channel': None,
                'processing': False
            }
        return self.user_sessions[user_id]
    
    def set_processing_state(self, guild_id, state, activity=""):
        """Set processing state for a guild"""
        # If bot is speaking and we're adding a new processing task, add to queue
        if self.speaking_states.get(guild_id, False) and state in ["recording", "transcribing", "processing", "thinking"]:
            if guild_id not in self.processing_queue:
                self.processing_queue[guild_id] = []
            self.processing_queue[guild_id].append(f"{state}: {activity}")
            self.queue_size[guild_id] = len(self.processing_queue[guild_id])
            self.logger.info(f"Added to queue for guild {guild_id}: {state} - {activity} (queue size: {self.queue_size[guild_id]})")
        else:
            # Normal state update
            self.processing_states[guild_id] = state
            if activity:
                self.current_activity[guild_id] = activity
            else:
                self.current_activity[guild_id] = ""
            self.logger.info(f"Processing state for guild {guild_id}: {state} - {activity}")
    
    def get_processing_state(self, guild_id):
        """Get current processing state for a guild"""
        return self.processing_states.get(guild_id, "idle")
    
    def get_current_activity(self, guild_id):
        """Get current activity description for a guild"""
        activity = self.current_activity.get(guild_id, "")
        
        # If bot is speaking and has queue, show queue info
        if self.speaking_states.get(guild_id, False) and self.queue_size.get(guild_id, 0) > 0:
            queue_info = f" (Queue: {self.queue_size[guild_id]} pending)"
            return activity + queue_info
        
        return activity
    
    def get_processing_queue(self, guild_id):
        """Get processing queue for a guild"""
        return self.processing_queue.get(guild_id, [])
    
    def get_queue_size(self, guild_id):
        """Get queue size for a guild"""
        return self.queue_size.get(guild_id, 0)
    
    def _get_guild_name(self, user, guild_id):
        """Get guild name from various sources"""
        if hasattr(user, 'guild') and user.guild:
            return user.guild.name
        elif hasattr(user, 'voice') and user.voice and user.voice.channel:
            return user.voice.channel.guild.name
        elif guild_id in self.active_voice_clients:
            vc = self.active_voice_clients[guild_id]
            if vc and vc.guild:
                return vc.guild.name
        return 'Unknown Guild'

    async def process_voice_message(self, user, text: str):
        """Process voice message from multi-user transcription"""
        try:
            user_id = user.id
            session = self.get_user_session(user_id)
            
            # Prevent duplicate processing
            if session['processing']:
                self.logger.warning(f"⚠️ Already processing for user {user.display_name}, skipping duplicate")
                return
            
            session['processing'] = True
            session['last_interaction'] = time.time()
            
            self.logger.info(f"🎯 Processing voice from {user.display_name}: {text}")
            
            # Get guild ID from voice channel context
            guild_id = 0
            if hasattr(user, 'guild') and user.guild:
                guild_id = user.guild.id
            elif hasattr(user, 'voice') and user.voice and user.voice.channel:
                guild_id = user.voice.channel.guild.id
            else:
                # Try to get guild from active voice clients
                for vc_guild_id, vc in self.active_voice_clients.items():
                    if vc and vc.guild:
                        guild_id = vc_guild_id
                        break
            
            # Set processing state
            self.set_processing_state(guild_id, "thinking", f"Processing message from {user.display_name}")
            
            # Send user voice message to dashboard
            await self.bot.send_conversation_update({
                'type': 'user_message',
                'user_id': str(user_id),
                'username': user.display_name,
                'guild_id': guild_id,
                'guild_name': self._get_guild_name(user, guild_id),
                'channel_id': user.voice.channel.id if hasattr(user, 'voice') and user.voice.channel else 0,
                'channel_name': user.voice.channel.name if hasattr(user, 'voice') and user.voice.channel else 'Voice Channel',
                'message': text,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
            # Process with LLM protection queue
            async with self.llm_semaphore:
                self.llm_queue_size += 1
                queue_pos = self.llm_queue_size
                self.logger.info(f"🛡️ LLM Queue position {queue_pos} for {user.display_name}")
                
                if queue_pos > 1:
                    self.logger.info(f"⏳ {user.display_name} waiting in queue (position {queue_pos})")
                    self.set_processing_state(guild_id, "waiting", f"Waiting in queue (position {queue_pos})")
                
                self.logger.info(f"🤖 Sending to LLM for user {user.display_name}")
                self.set_processing_state(guild_id, "thinking", "Generating response with AI")
                result = await self.conversation_handler.process_interaction(
                    user_id=str(user_id),
                    guild_id=guild_id,
                    message_content=text,
                    interaction_type="voice",
                    username=user.display_name
                )
                self.logger.info(f"✅ LLM response received for user {user.display_name}")
                self.llm_queue_size -= 1
                
                # Update status to show LLM response received
                if guild_id:
                    self.set_processing_state(guild_id, "thinking", "LLM response received, preparing response")
            
            if result['should_respond'] and result['response']:
                # Send bot voice response to dashboard
                await self.bot.send_conversation_update({
                    'type': 'bot_response',
                    'user_id': str(user_id),
                    'username': user.display_name,
                    'guild_id': guild_id,
                    'guild_name': self._get_guild_name(user, guild_id),
                    'channel_id': user.voice.channel.id if hasattr(user, 'voice') and user.voice.channel else 0,
                    'channel_name': user.voice.channel.name if hasattr(user, 'voice') and user.voice.channel else 'Voice Channel',
                    'response': result['response'],
                    'timestamp': datetime.datetime.now().isoformat()
                })
                
                await self.send_voice_response(user, result['response'])

        except Exception as e:
            self.logger.error(f"Error processing voice message for {user.display_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if user_id in self.user_sessions:
                self.user_sessions[user_id]['processing'] = False
                self.logger.info(f"🏁 Finished processing for user {user.display_name}")
                
                # Reset processing state
                guild_id = 0
                if hasattr(user, 'guild') and user.guild:
                    guild_id = user.guild.id
                elif hasattr(user, 'voice') and user.voice and user.voice.channel:
                    guild_id = user.voice.channel.guild.id
                else:
                    # Try to get guild from active voice clients
                    for vc_guild_id, vc in self.active_voice_clients.items():
                        if vc and vc.guild:
                            guild_id = vc_guild_id
                            break
                
                if guild_id:
                    self.set_processing_state(guild_id, "idle", "")

    async def interrupt_bot(self):
        """Interrupt the bot's current speaking/processing"""
        try:
            interrupted_count = 0
            interrupt_info = {
                'timestamp': datetime.datetime.now().isoformat(),
                'processing_states': {},
                'interrupted_audio': False,
                'interrupted_processing': False
            }
            
            # Stop all active voice clients
            for guild_id, vc in self.active_voice_clients.items():
                if vc and vc.is_connected():
                    # Track what was being processed
                    current_state = self.get_processing_state(guild_id)
                    interrupt_info['processing_states'][guild_id] = current_state
                    
                    # Stop any current audio playback
                    if vc.is_playing():
                        vc.stop()
                        interrupt_info['interrupted_audio'] = True
                        self.logger.info(f"Stopped audio playback in guild {guild_id}")
                    
                    # Check if we were processing something
                    if current_state != 'idle':
                        interrupt_info['interrupted_processing'] = True
                    
                    # Reset processing state
                    self.set_processing_state(guild_id, "idle", "")
                    interrupted_count += 1
                    self.logger.info(f"Interrupted bot in guild {guild_id}")
            
            # Clear all user processing flags to allow new input
            for user_id, session in self.user_sessions.items():
                if session.get('processing', False):
                    session['processing'] = False
                    self.logger.info(f"Cleared processing flag for user {user_id}")
            
            # Reset LLM queue and semaphore to clear any stuck processing
            if self.llm_queue_size > 0:
                self.logger.info(f"Resetting LLM queue size from {self.llm_queue_size} to 0")
                self.llm_queue_size = 0
            
            # Force reset the semaphore to clear any stuck LLM processing
            try:
                self.logger.info("Force resetting LLM semaphore to clear stuck processing")
                self.llm_semaphore = asyncio.Semaphore(1)
                self.logger.info("Created new LLM semaphore - cleared any stuck processing")
            except Exception as e:
                self.logger.warning(f"Could not reset LLM semaphore: {e}")
            
            # Ensure VoiceSink is enabled and ready to listen
            for guild_id, vc in self.active_voice_clients.items():
                if vc and vc.is_connected():
                    # Make sure bot is not deafened
                    if hasattr(vc, 'self_deaf') and vc.self_deaf:
                        await vc.guild.change_voice_state(channel=vc.channel, self_deaf=False)
                        self.logger.info(f"Undeafened bot in guild {guild_id} after interrupt")
                    
                    # Completely recreate VoiceSink to ensure it's working
                    try:
                        self.logger.info(f"Recreating VoiceSink for guild {guild_id}")
                        
                        # Stop the old sink
                        if hasattr(vc, 'sink') and vc.sink:
                            vc.sink.cleanup()
                            self.logger.info(f"Cleaned up old VoiceSink for guild {guild_id}")
                        
                        # Create new VoiceSink
                        new_sink = VoiceSink(self, self.bot)
                        self.logger.info(f"Created new VoiceSink for guild {guild_id}")
                        
                        # Stop listening to old sink and start listening to new one
                        vc.stop_listening()
                        await asyncio.sleep(0.1)  # Small delay
                        vc.listen(new_sink)
                        vc.sink = new_sink
                        
                        self.logger.info(f"Reinitialized VoiceSink for guild {guild_id}")
                        
                    except Exception as e:
                        self.logger.error(f"Error recreating VoiceSink for guild {guild_id}: {e}")
                        # Fallback to just enabling the existing sink
                        if hasattr(vc, 'sink') and vc.sink:
                            vc.sink.decode = True
                            self.logger.info(f"Fallback: Ensured existing VoiceSink is enabled for guild {guild_id}")
            
            # Add a small delay to ensure everything is properly reset
            await asyncio.sleep(0.5)
            
            # Final verification - check if VoiceSink is properly enabled
            for guild_id, vc in self.active_voice_clients.items():
                if vc and vc.is_connected() and hasattr(vc, 'sink') and vc.sink:
                    if not vc.sink.decode:
                        self.logger.warning(f"VoiceSink still disabled for guild {guild_id}, forcing enable")
                        vc.sink.decode = True
                    self.logger.info(f"Final VoiceSink state for guild {guild_id}: decode={vc.sink.decode}")
                    self.logger.info(f"VoiceSink user_recordings count: {len(vc.sink.user_recordings)}")
                    self.logger.info(f"VoiceSink is properly initialized and ready to listen")
            
            if interrupted_count > 0:
                # Send immediate dashboard update
                await self.bot.send_dashboard_update()
                self.bot.logger.info("Dashboard update sent successfully")
                self.logger.info("Bot interrupted successfully - ready to listen again")
                return True
            else:
                self.logger.warning("No active voice clients to interrupt")
                return False
                
        except Exception as e:
            self.logger.error(f"Error interrupting bot: {e}")
            return False

    async def send_voice_response(self, user, response_text):
        """Send voice response to connected voice channel"""
        try:
            # Use the first available active voice client (since we're connected)
            voice_client = None
            voice_channel = None
            
            for guild_id, vc in self.active_voice_clients.items():
                if vc and vc.is_connected():
                    voice_client = vc
                    voice_channel = vc.channel
                    break
            
            if not voice_client:
                self.logger.warning("No active voice client found")
                return
            
            self.logger.info(f"Sending voice response to {voice_channel.name}")
            
            # Set speaking state
            guild_id = voice_channel.guild.id
            self.set_processing_state(guild_id, "speaking", "Generating and playing speech")
            
            # Generate and play speech
            wav_file = await self.piper.generate_speech(response_text)
            if wav_file:
                await self.play_audio_non_blocking(voice_client, wav_file)
            
            # Send text response to voice channel's text chat or related channel
            try:
                guild = voice_channel.guild
                text_channel = None
                
                # Method 1: Check if voice channel has text chat permissions
                if hasattr(voice_channel, 'send') and voice_channel.permissions_for(guild.me).send_messages:
                    text_channel = voice_channel
                    self.logger.info(f"Using voice channel text chat: {voice_channel.name}")
                
                # Method 2: Find channel with similar name (e.g., "general" voice → "general" text)
                if not text_channel:
                    voice_name = voice_channel.name.lower()
                    for channel in guild.text_channels:
                        if (channel.name.lower() == voice_name or 
                            voice_name in channel.name.lower() or
                            channel.name.lower() in voice_name) and \
                           channel.permissions_for(guild.me).send_messages:
                            text_channel = channel
                            self.logger.info(f"Found matching text channel: {channel.name}")
                            break
                
                # Method 3: Find any general/main text channel
                if not text_channel:
                    priority_names = ['general', 'main', 'chat', 'bot', 'commands']
                    for name in priority_names:
                        for channel in guild.text_channels:
                            if name in channel.name.lower() and channel.permissions_for(guild.me).send_messages:
                                text_channel = channel
                                self.logger.info(f"Using priority text channel: {channel.name}")
                                break
                        if text_channel:
                            break
                
                # Method 4: Use first available text channel
                if not text_channel:
                    for channel in guild.text_channels:
                        if channel.permissions_for(guild.me).send_messages:
                            text_channel = channel
                            self.logger.info(f"Using first available text channel: {channel.name}")
                            break
                
                if text_channel:
                    await text_channel.send(f"🎤 **{user.display_name}**: {response_text}")
                    self.logger.info(f"Sent text response to #{text_channel.name}")
                else:
                    self.logger.warning("No suitable text channel found for response")
                    
            except Exception as text_error:
                self.logger.error(f"Error sending text message: {text_error}")

        except Exception as e:
            self.logger.error(f"Error sending voice response: {e}")

    async def play_audio_non_blocking(self, voice_client, wav_file):
        """Play audio without blocking other users"""
        try:
            if not voice_client or not voice_client.is_connected():
                return
            
            # Check if already playing - if so, queue or skip
            if voice_client.is_playing():
                self.logger.info("Voice client busy, skipping audio")
                try:
                    os.remove(wav_file)
                except:
                    pass
                return
                
            source = discord.FFmpegPCMAudio(wav_file)
            voice_client.play(source, after=lambda e: self._audio_finished(wav_file, e))
            self.logger.info(f"Started playing audio: {wav_file}")
            
            # Update status to show audio is now playing
            for vc_guild_id, vc in self.active_voice_clients.items():
                if vc and vc.is_connected() and vc == voice_client:
                    self.set_processing_state(vc_guild_id, "speaking", "Playing audio response")
                    break
            
            # Wait for audio to finish playing
            while voice_client.is_playing():
                await asyncio.sleep(0.1)
            self.logger.info("Audio playback completed")
            
            # Clear speaking state when audio finishes and process queue
            for vc_guild_id, vc in self.active_voice_clients.items():
                if vc and vc.is_connected() and vc == voice_client:
                    # Process next item in queue if available
                    if vc_guild_id in self.processing_queue and self.processing_queue[vc_guild_id]:
                        next_item = self.processing_queue[vc_guild_id].pop(0)
                        self.queue_size[vc_guild_id] = len(self.processing_queue[vc_guild_id])
                        self.logger.info(f"Processing next queue item for guild {vc_guild_id}: {next_item}")
                        # Set the state from the queue item
                        if ": " in next_item:
                            state, activity = next_item.split(": ", 1)
                            self.set_processing_state(vc_guild_id, state, activity)
                        else:
                            self.set_processing_state(vc_guild_id, "processing", next_item)
                    else:
                        # No queue items, set to idle
                        self.set_processing_state(vc_guild_id, "idle", "")
                        if vc_guild_id in self.processing_queue:
                            del self.processing_queue[vc_guild_id]
                        if vc_guild_id in self.queue_size:
                            del self.queue_size[vc_guild_id]
                    self.logger.info(f"Cleared speaking state for guild {vc_guild_id}")
                    break
            
        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")
            try:
                os.remove(wav_file)
            except:
                pass

    def _audio_finished(self, wav_file, error):
        """Cleanup after audio finishes"""
        if error:
            self.logger.error(f"Audio playback error: {error}")
        try:
            os.remove(wav_file)
        except Exception as e:
            self.logger.error(f"Error removing audio file: {e}")

    async def handle_text(self, user, text: str):
        """Legacy method - redirects to new multi-user processing"""
        await self.process_voice_message(user, text)

    async def play_audio(self, voice_client, wav_file):
        try:
            if not voice_client or not voice_client.is_connected():
                return
                
            source = discord.FFmpegPCMAudio(wav_file)
            voice_client.play(source, after=lambda e: print(f'Player error: {e}') if e else None)
            
            while voice_client.is_playing():
                await asyncio.sleep(0.1)
                
            try:
                os.remove(wav_file)
            except Exception as e:
                self.logger.error(f"Error removing audio file: {e}")
                
        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")
            import traceback
            traceback.print_exc()

    @commands.command()
    async def vc(self, ctx):
        if not ctx.author.voice:
            await ctx.send("You need to be in a voice channel!")
            return

        try:
            self.logger.info(f"Joining channel: {ctx.author.voice.channel}")
            
            await self.piper.initialize()
            self.logger.info("Piper initialized")
            
            # Disconnect any existing connection first
            if ctx.voice_client:
                await ctx.voice_client.disconnect()
                await asyncio.sleep(1)
            
            # Connect with new discord.py version (includes 4006 fix)
            vc = await ctx.author.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
            self.logger.info("Connected to voice channel")
            
            # Track voice client for multi-user management
            guild_id = ctx.guild.id
            self.active_voice_clients[guild_id] = vc
            
            # Wait for connection to stabilize
            await asyncio.sleep(2)
            
            # Verify connection before proceeding
            if not vc.is_connected():
                raise Exception("Voice connection failed to establish properly")
            
            sink = VoiceSink(self, self.bot)
            self.logger.info("Created multi-user voice sink")
            
            vc.listen(sink)
            self.logger.info("Started listening for multiple users")
            
            # Wait a moment for voice connection to stabilize
            await asyncio.sleep(0.5)
            
            # Store sink reference for later control (after listen is called)
            vc.sink = sink
            
            # Send immediate dashboard update
            self.logger.info("Sending immediate dashboard update after voice connection")
            await self.bot.send_dashboard_update()
            
            await ctx.send("🎙️ Multi-user voice chat ready! Everyone can speak!")
            
        except Exception as e:
            self.logger.error(f"Error joining voice: {e}")
            if ctx.voice_client:
                await ctx.voice_client.disconnect()
            await ctx.send("❌ Failed to join voice channel")

    @commands.command()
    async def stop(self, ctx):
        guild_id = ctx.guild.id
        if ctx.voice_client:
            # Clean up voice client tracking
            if guild_id in self.active_voice_clients:
                del self.active_voice_clients[guild_id]
            
            # Clear user sessions for this guild
            users_to_remove = [user_id for user_id, session in self.user_sessions.items() 
                             if session.get('voice_channel') and session['voice_channel'].guild.id == guild_id]
            for user_id in users_to_remove:
                del self.user_sessions[user_id]
            
            await ctx.voice_client.disconnect()
            
            # Send immediate dashboard update
            self.logger.info("Sending immediate dashboard update after voice disconnection")
            await self.bot.send_dashboard_update()
            
            await ctx.send("👋 Multi-user voice chat stopped!")
        else:
            await ctx.send("❌ Not in a voice channel!")

    @commands.command()
    async def deafen(self, ctx):
        """Deafen the bot (stop it from hearing voice)"""
        try:
            # Check if bot is in voice
            if not ctx.voice_client:
                await ctx.send("❌ Bot is not in a voice channel!")
                return
            
            # Deafen the bot
            await ctx.guild.change_voice_state(channel=ctx.voice_client.channel, self_deaf=True)
            self.logger.info(f"Bot deafened in guild {ctx.guild.id}")
            
            # Disable VoiceSink to stop listening
            if hasattr(ctx.voice_client, 'sink') and ctx.voice_client.sink:
                ctx.voice_client.sink.decode = False
                self.logger.info("VoiceSink disabled - bot stopped listening")
            else:
                self.logger.warning("VoiceSink not found or not accessible - deafen may not stop listening")
            
            # Send immediate dashboard update
            await self.bot.send_dashboard_update()
            
            await ctx.send("🔇 Bot deafened! I can't hear voice anymore.")
            
        except Exception as e:
            self.logger.error(f"Error deafening bot: {e}")
            await ctx.send("❌ Failed to deafen bot")

    @commands.command()
    async def undeafen(self, ctx):
        """Undeafen the bot (allow it to hear voice again)"""
        try:
            # Check if bot is in voice
            if not ctx.voice_client:
                await ctx.send("❌ Bot is not in a voice channel!")
                return
            
            # Undeafen the bot
            await ctx.guild.change_voice_state(channel=ctx.voice_client.channel, self_deaf=False)
            self.logger.info(f"Bot undeafened in guild {ctx.guild.id}")
            
            # Re-enable VoiceSink to resume listening
            if hasattr(ctx.voice_client, 'sink') and ctx.voice_client.sink:
                ctx.voice_client.sink.decode = True
                self.logger.info("VoiceSink enabled - bot resumed listening")
            else:
                self.logger.warning("VoiceSink not found or not accessible - undeafen may not resume listening")
            
            # Send immediate dashboard update
            await self.bot.send_dashboard_update()
            
            await ctx.send("👂 Bot undeafened! I can hear voice again.")
            
        except Exception as e:
            self.logger.error(f"Error undeafening bot: {e}")
            await ctx.send("❌ Failed to undeafen bot")

    @commands.command()
    async def die(self, ctx):
        if ctx.voice_client:
            ctx.voice_client.stop()
        await ctx.send("💤 Shutting down...")
        await ctx.bot.close()

class PiperTTS:
    def __init__(self, 
                 piper_path: str = 'piper/piper.exe',
                 model_path: str = 'piper/en_US-ryari-high.onnx',
                 model_config: str = 'piper/en_US-ryari-high.onnx.json',
                 output_dir: str = 'output',
                 timeout: int = 10):
        self.piper_path = Path(piper_path)
        self.model_path = Path(model_path)
        self.model_config = Path(model_config)
        self.output_dir = Path(output_dir)
        self.timeout = timeout
        
        self.process: Optional[subprocess.Popen] = None
        self.initialized = False
        self.lock = asyncio.Lock()
        self.current_generation: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger('PiperTTS')
        self.logger.setLevel(logging.INFO)
        
        self.output_dir.mkdir(exist_ok=True)
        
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_generation_time': 0
        }

    async def verify_files(self) -> bool:
        required_files = [
            (self.piper_path, "Piper executable"),
            (self.model_path, "Model file"),
            (self.model_config, "Model configuration")
        ]
        
        for file_path, description in required_files:
            if not file_path.exists():
                self.logger.error(f"Missing {description} at {file_path}")
                return False
        return True

    async def initialize(self) -> bool:
        if self.initialized:
            return True
            
        try:
            async with self.lock:
                if not await self.verify_files():
                    return False
                
                if self.process and self.process.poll() is None:
                    self.process.terminate()
                    await asyncio.sleep(0.1)
                    
                self.process = subprocess.Popen(
                    [
                        str(self.piper_path),
                        '-m', str(self.model_path),
                        '-c', str(self.model_config),
                        '--json-input'
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                await asyncio.sleep(0.1)
                if self.process.poll() is not None:
                    stderr = self.process.stderr.read()
                    self.logger.error(f"Piper failed to start: {stderr}")
                    return False
                
                self.initialized = True
                self.logger.info("Piper initialized successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False

    async def generate_speech(self, text: str) -> Optional[str]:
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            if not self.initialized and not await self.initialize():
                return None
                
            async with self.lock:
                output_file = str(self.output_dir / f'output_{int(time.time() * 1000)}.wav')
                input_json = {
                    'text': text,
                    'output_file': output_file,
                    'length_scale': 1.0,
                    'noise_scale': 0.667,
                    'noise_w': 0.8
                }
                
                process = await asyncio.create_subprocess_exec(
                    str(self.piper_path),
                    '-m', str(self.model_path),
                    '-c', str(self.model_config),
                    '--json-input',
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(json.dumps(input_json).encode() + b'\n'),
                        timeout=self.timeout
                    )
                    
                    if process.returncode != 0:
                        self.logger.error(f"Piper process failed: {stderr.decode()}")
                        self.stats['failed_requests'] += 1
                        return None
                    
                    if not os.path.exists(output_file):
                        self.logger.error("Output file was not created")
                        self.stats['failed_requests'] += 1
                        return None
                        
                    self.stats['successful_requests'] += 1
                    generation_time = time.time() - start_time
                    self.stats['average_generation_time'] = (
                        (self.stats['average_generation_time'] * 
                         (self.stats['successful_requests'] - 1) +
                         generation_time) / self.stats['successful_requests']
                    )
                    
                    self.logger.info(f"Generated speech in {generation_time:.2f}s: {output_file}")
                    return output_file
                    
                except asyncio.TimeoutError:
                    self.logger.error("Piper process timed out")
                    process.kill()
                    self.stats['failed_requests'] += 1
                    return None
                    
                except Exception as e:
                    self.logger.error(f"Generation error: {str(e)}")
                    self.stats['failed_requests'] += 1
                    return None
                    
        except Exception as e:
            self.logger.error(f"Unexpected error in generate_speech: {str(e)}")
            self.stats['failed_requests'] += 1
            return None

    async def cleanup(self):
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(asyncio.sleep(0)),
                        timeout=2
                    )
                except asyncio.TimeoutError:
                    self.process.kill()
                    
            for file in self.output_dir.glob('*.wav'):
                try:
                    os.remove(file)
                except Exception as e:
                    self.logger.error(f"Failed to remove file {file}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
        finally:
            self.initialized = False
            self.process = None

class AudioProcessor:
    @staticmethod
    def get_audio_info(file_path):
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)
            
            duration = float(data['format']['duration'])
            stream = next((s for s in data['streams'] if s['codec_type'] == 'audio'), None)
            sample_rate = int(stream['sample_rate']) if stream else 48000
            channels = int(stream['channels']) if stream else 1
            
            return duration, sample_rate, channels
        except Exception as e:
            print(f"Error getting audio info: {e}")
            return 5.0, 48000, 1

    @staticmethod
    def fast_convert_and_analyze(input_file, output_file):
        try:
            duration, sample_rate, channels = AudioProcessor.get_audio_info(input_file)

            cmd = [
                'ffmpeg',
                '-i', input_file,
                '-vn',
                '-ar', '48000',
                '-ac', '1',
                '-f', 's16le',
                '-acodec', 'pcm_s16le',
                'pipe:1',
                '-y'
            ]
            
            process = subprocess.run(cmd, capture_output=True)
            audio_data = np.frombuffer(process.stdout, dtype=np.int16)
            
            amplitudes = np.abs(audio_data)
            segments = np.array_split(amplitudes, 256)
            waveform = np.array([np.max(segment) if len(segment) > 0 else 0 for segment in segments])
            
            max_val = np.max(waveform)
            if max_val > 0:
                waveform = (waveform / max_val * 255).astype(np.uint8)
                window_size = 3
                smoothed = np.convolve(waveform, np.ones(window_size)/window_size, mode='same')
                waveform = smoothed.astype(np.uint8)
                waveform = np.maximum(waveform, 10)
            else:
                waveform = np.full(256, 128, dtype=np.uint8)
            
            waveform_base64 = base64.b64encode(bytes(waveform.tolist())).decode('utf-8')

            subprocess.run([
                'ffmpeg',
                '-i', input_file,
                '-c:a', 'libopus',
                '-b:a', '64k',
                output_file
            ], check=True)

            return waveform_base64, duration

        except Exception as e:
            print(f"Error in fast convert and analyze: {e}")
            return base64.b64encode(bytes([128] * 256)).decode('utf-8'), 5.0

class VoiceSink(voice_recv.AudioSink):
    def __init__(self, cog, bot):
        super().__init__()
        self.cog = cog
        self.bot = bot
        self.decode = True
        
        # Initialize Faster-Whisper (CPU mode for stability)
        try:
            # Force CPU mode until PyTorch CUDA issues are resolved
            self.whisper_model = WhisperModel(
                "base",  # Good balance of speed and accuracy
                device="cpu",
                compute_type="int8"
            )
            print(f"🎤 Faster-Whisper initialized on CPU (stable mode)")
        except Exception as e:
            print(f"❌ Faster-Whisper failed: {e}")
            self.whisper_model = None
            
        # Multi-user recording state
        self.user_recordings = {}  # user_id -> recording data
        self.output_dir = Path('recordings')
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger('VoiceSink')
        
        # Voice Activity Detection settings
        self.SILENCE_THRESHOLD_MS = 1500  # Stop recording after 1.5s silence
        self.MIN_RECORDING_MS = 500       # Minimum recording duration
        self.AMPLITUDE_THRESHOLD = 100    # Minimum amplitude to detect speech
        
        print("Multi-user VoiceSink initialized")

    def wants_opus(self) -> bool:
        return False

    def cleanup(self):
        self.logger.info("VoiceSink cleanup")
        self.decode = False
        self.user_recordings = {}

    def is_speaking(self, audio_data):
        """Detect if user is speaking based on amplitude"""
        max_amplitude = np.max(np.abs(audio_data))
        return max_amplitude > self.AMPLITUDE_THRESHOLD

    async def finalize_recording(self, user_id):
        """Save and queue recording for transcription"""
        if user_id not in self.user_recordings:
            return
            
        recording_data = self.user_recordings[user_id]
        buffer = recording_data['buffer']
        
        # Check minimum duration
        duration_ms = len(buffer) * 20  # 20ms per chunk
        if duration_ms < self.MIN_RECORDING_MS:
            self.logger.debug(f"Recording too short for user {user_id}: {duration_ms}ms")
            self.user_recordings[user_id] = self._init_user_recording()
            return
        
        # Save audio file
        filename = f"temp_{user_id}_{int(time.time() * 1000)}.wav"
        filepath = self.output_dir / filename
        
        try:
            # Combine all audio chunks
            combined_audio = b''.join(buffer)
            audio_np = np.frombuffer(combined_audio, dtype=np.int16)
            
            # Save as WAV file
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(48000)  # 48kHz
                wav_file.writeframes(combined_audio)
            
            self.logger.info(f"Saved recording for user {user_id}: {filename} ({duration_ms}ms)")
            
            # Queue for async transcription (non-blocking)
            asyncio.run_coroutine_threadsafe(
                self.transcribe_and_cleanup(str(filepath), user_id),
                self.bot.loop
            )
            
        except Exception as e:
            self.logger.error(f"Error saving recording for user {user_id}: {e}")
        
        # Reset user recording state
        self.user_recordings[user_id] = self._init_user_recording()

    def _init_user_recording(self):
        """Initialize recording state for a user"""
        return {
            'buffer': [],
            'last_activity': time.time(),
            'recording': False,
            'start_time': None
        }

    async def transcribe_and_cleanup(self, filepath, user_id):
        """Transcribe audio file and clean up (async)"""
        try:
            if not self.whisper_model:
                self.logger.error("Whisper model not available")
                return
            
            # Set transcribing state
            guild_id = 0
            if hasattr(self.cog, 'active_voice_clients'):
                for vc_guild_id, vc in self.cog.active_voice_clients.items():
                    if vc and vc.is_connected():
                        guild_id = vc_guild_id
                        break
            
            if guild_id and hasattr(self.cog, 'set_processing_state'):
                self.cog.set_processing_state(guild_id, "transcribing", "Converting speech to text")
            
            # Transcribe audio
            segments, info = self.whisper_model.transcribe(filepath, beam_size=5)
            transcription = " ".join([segment.text for segment in segments]).strip()
            
            if transcription:
                self.logger.info(f"Transcription for user {user_id}: {transcription}")
                
                # Get user object - try cache first, then API
                user = self.bot.get_user(user_id)
                if user:
                    self.logger.info(f"Found user {user.display_name} in cache, processing message...")
                else:
                    self.logger.info(f"User {user_id} not in cache, fetching from Discord API...")
                    try:
                        user = await self.bot.fetch_user(user_id)
                        if user:
                            self.logger.info(f"Fetched user {user.display_name} from API, processing message...")
                        else:
                            self.logger.error(f"Could not fetch user {user_id} from Discord API")
                            return
                    except Exception as fetch_error:
                        self.logger.error(f"Error fetching user {user_id}: {fetch_error}")
                        return
                
                # Process the message (only called once now)
                if user:
                    # Set processing state for voice message processing
                    guild_id = user.guild.id if hasattr(user, 'guild') and user.guild else 0
                    if guild_id and hasattr(self.cog, 'set_processing_state'):
                        self.cog.set_processing_state(guild_id, "processing", f"Processing voice from {user.display_name}")
                    
                    await self.cog.process_voice_message(user, transcription)
            
        except Exception as e:
            self.logger.error(f"Transcription error for user {user_id}: {e}")
        
        finally:
            # Clear transcribing state
            if guild_id and hasattr(self.cog, 'set_processing_state'):
                self.cog.set_processing_state(guild_id, "idle", "")
                self.logger.info(f"Cleared transcribing state for guild {guild_id}")
            
            # Clean up file
            try:
                os.remove(filepath)
                self.logger.debug(f"Cleaned up file: {filepath}")
            except Exception as e:
                self.logger.error(f"Error removing file {filepath}: {e}")

    def write(self, user, data: voice_recv.VoiceData):
        try:
            # Check if VoiceSink is disabled (deafened)
            if not self.decode:
                return
            
            if user is None or data.pcm is None:
                return
            
            user_id = user.id
            
            # Initialize user recording if not exists
            if user_id not in self.user_recordings:
                self.user_recordings[user_id] = self._init_user_recording()
            
            recording = self.user_recordings[user_id]
            
            # Only record if we're in recording state (set by Discord speaking events)
            if recording['recording']:
                try:
                    audio_np = np.frombuffer(data.pcm, dtype=np.int16)
                    audio_stereo = audio_np.reshape(-1, 2)
                    audio_mono = audio_stereo.mean(axis=1).astype(np.int16)
                    
                    recording['buffer'].append(audio_mono.tobytes())
                    recording['last_activity'] = time.time()
                    
                except Exception as e:
                    self.logger.error(f"Error processing audio: {e}")
                
        except Exception as e:
            self.logger.error(f"Error in write: {e}")

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_start(self, member):
        """Discord detected user started speaking (push-to-talk pressed)"""
        try:
            # Check if VoiceSink is disabled (deafened)
            if not self.decode:
                return
                
            user_id = member.id
            if user_id not in self.user_recordings:
                self.user_recordings[user_id] = self._init_user_recording()
            
            recording = self.user_recordings[user_id]
            if not recording['recording']:
                recording['recording'] = True
                recording['start_time'] = time.time()
                recording['buffer'] = []  # Clear any old data
                self.logger.info(f"🎤 Started recording for {member.display_name}")
                
                # Set recording state
                guild_id = member.guild.id if hasattr(member, 'guild') and member.guild else 0
                if guild_id and hasattr(self.cog, 'set_processing_state'):
                    self.cog.set_processing_state(guild_id, "recording", f"Recording from {member.display_name}")
                
        except Exception as e:
            self.logger.error(f"Error in speaking start: {e}")

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_stop(self, member):
        """Discord detected user stopped speaking (push-to-talk released)"""
        try:
            # Check if VoiceSink is disabled (deafened)
            if not self.decode:
                return
                
            user_id = member.id
            if user_id in self.user_recordings and self.user_recordings[user_id]['recording']:
                self.logger.info(f"🎤 Stopped recording for {member.display_name}")
                
                # Clear recording state
                guild_id = member.guild.id if hasattr(member, 'guild') and member.guild else 0
                if guild_id and hasattr(self.cog, 'set_processing_state'):
                    self.cog.set_processing_state(guild_id, "idle", "")
                    self.logger.info(f"Cleared recording state for guild {guild_id}")
                
                # Immediately finalize recording when user releases push-to-talk
                asyncio.run_coroutine_threadsafe(
                    self.finalize_recording(user_id),
                    self.bot.loop
                )
                
        except Exception as e:
            self.logger.error(f"Error in speaking stop: {e}")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('discord_bot.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Fix Windows Unicode issues
    import sys
    if sys.platform == 'win32':
        import os
        os.environ['PYTHONIOENCODING'] = 'utf-8'

def main():
    setup_logging()
    logger = logging.getLogger('main')
    
    # Pre-load models before Discord connection
    logger.info("🔄 Pre-loading AI models...")
    
    # Initialize memory store (loads SentenceTransformer on GPU)
    logger.info("📚 Loading memory system...")
    memory_store = EnhancedMemoryStore()
    
    # Pre-load Faster-Whisper (CPU mode for stability)
    logger.info("🎤 Loading Faster-Whisper...")
    try:
        # Use CPU mode until PyTorch CUDA issues are resolved
        whisper_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )
        logger.info("✅ Faster-Whisper loaded on CPU (stable mode)")
    except Exception as e:
        logger.error(f"❌ Faster-Whisper loading failed: {e}")
    
    # Pre-warm Ollama model
    logger.info("🤖 Pre-warming Ollama model...")
    try:
        import asyncio
        async def warm_ollama():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: ollama.chat(
                    model='hf.co/subsectmusic/qwriko3-4b-instruct-2507:Q4_K_M',
                    messages=[{'role': 'user', 'content': 'warmup'}],
                    options={'num_predict': 1}
                )
            )
        asyncio.run(warm_ollama())
        logger.info("✅ Ollama model warmed up")
    except Exception as e:
        logger.warning(f"⚠️ Ollama warmup failed: {e}")
    
    logger.info("✅ All models loaded! Starting Discord bot...")
    
    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True
    intents.guilds = True

    bot = Bot(command_prefix='!', intents=intents, memory_store=memory_store)

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return

        if message.channel.id not in bot.channel_contexts:
            bot.channel_contexts[message.channel.id] = ChannelContext()

        context = bot.channel_contexts[message.channel.id].get_context()

        # Scrub the bot's username from the message content
        scrubbed_content = bot.scrub_bot_username(message.content)

        is_direct = (bot.user in message.mentions or (
            message.reference and 
            message.reference.resolved and 
            message.reference.resolved.author == bot.user
        ))

        # Always check for commands first
        await bot.process_commands(message)
        
        # Then handle direct mentions/replies if it's not a command
        if is_direct and not message.content.startswith('!'):
            logger.info("\nProcessing direct mention...")
            
            # Send user message to dashboard
            await bot.send_conversation_update({
                'type': 'user_message',
                'user_id': str(message.author.id),
                'username': message.author.display_name,
                'guild_id': message.guild.id,
                'guild_name': message.guild.name,
                'channel_id': message.channel.id,
                'channel_name': message.channel.name,
                'message': scrubbed_content,
                'timestamp': message.created_at.isoformat()
            })
            
            response = await bot.conversation_handler.process_interaction(
                user_id=str(message.author),
                guild_id=message.guild.id,
                message_content=scrubbed_content,
                interaction_type="direct_mention",
                context=context,
                username=message.author.display_name
            )
            if response['response']:
                # Send bot response to dashboard
                await bot.send_conversation_update({
                    'type': 'bot_response',
                    'user_id': str(message.author.id),
                    'username': message.author.display_name,
                    'guild_id': message.guild.id,
                    'guild_name': message.guild.name,
                    'channel_id': message.channel.id,
                    'channel_name': message.channel.name,
                    'response': response['response'],
                    'timestamp': datetime.datetime.now().isoformat()
                })
                
                await bot.process_message(message, response['response'], use_tts=True)

    async def setup_hook():
        await bot.add_cog(Testing(bot, bot.piper))

    bot.setup_hook = setup_hook

    try:
        logger.info("Starting bot...")
        logger.info("Make sure:")
        logger.info("1. Piper files are in 'piper' directory")
        logger.info("2. 'recordings' and 'output' directories exist")
        logger.info("3. Ollama is running with: ollama run llama3.1:8b")
        logger.info("\nStarting bot now...")
        bot.run(DISCORD_BOT_TOKEN)
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()