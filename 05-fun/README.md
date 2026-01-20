# JetBrain - AI Chatbot with Persistent Memory

A Flask-based AI chatbot running on Jetson with vision capabilities and persistent memory.

## Features

- **Chat with LLM** - Powered by Ollama (gemma3, llama3, etc.)
- **Persistent Memory** - Remembers things across conversations via `.jetbrain` file
- **Slash Commands** - Control memory, switch models, check status
- **Image Analysis** - Upload images and ask questions about them
- **Chat History** - SQLite database saves all conversations
- **Dark Theme** - Clean UI with dot grid background
- **GPU Accelerated** - Uses CUDA for fast vision inference

## Quick Start

### 1. Make sure Ollama is running

```bash
# Start Ollama server
ollama serve

# Pull a model (in another terminal)
ollama pull gemma3:1b
```

### 2. Install Flask (if needed)

```bash
source ../setup_env.sh
pip install flask
```

### 3. Run JetBrain

```bash
python app.py
```

### 4. Open in browser

Navigate to: **http://localhost:5000**

---

## Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/memory` | Display current memory |
| `/remember <text>` | Save something to memory |
| `/forget <text>` | Remove something from memory |
| `/clear <category>` | Clear a memory category |
| `/status` | Check system status (GPU, Ollama, etc.) |
| `/model <name>` | Switch LLM model |

### Memory Categories

- **facts** - Things about you
- **preferences** - Your preferences
- **projects** - Current projects
- **notes** - General notes

### Examples

```
/remember fact: I work at NVIDIA
/remember preference: I prefer Python over JavaScript
/remember project: Building a robot arm with ROS2
/remember note: Meeting with team on Friday

/memory                    # Show all memories
/forget Building a robot arm with ROS2
/clear notes              # Clear all notes
/clear all                # Clear everything

/model llama3.2:1b        # Switch to different model
/status                   # Check system status
```

---

## How Memory Works

JetBrain stores memories in a `.jetbrain` file (JSON format). This file persists across:
- Different chat sessions
- Server restarts
- New conversations

Every time JetBrain responds, it loads the memory and includes it in its context, so it always knows what you've told it.

### Memory File Location
```
05-fun/.jetbrain
```

### Memory Format
```json
{
  "facts": [
    "I work at NVIDIA",
    "My favorite language is Python"
  ],
  "preferences": [
    "I prefer dark themes",
    "I like concise responses"
  ],
  "projects": [
    "Building a robot arm with ROS2"
  ],
  "notes": [
    "Meeting on Friday"
  ]
}
```

---

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Flask application |
| `templates/index.html` | Chat UI |
| `.jetbrain` | Persistent memory (JSON) |
| `jetbrain.db` | Chat history (SQLite) |
| `uploads/` | Uploaded images |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main chat interface |
| `/api/chat` | POST | Send message, get response |
| `/api/conversations` | GET | List all conversations |
| `/api/conversations` | POST | Create new conversation |
| `/api/conversations/<id>` | DELETE | Delete a conversation |
| `/api/conversations/<id>/messages` | GET | Get messages |
| `/api/memory` | GET | Get current memory |
| `/api/status` | GET | Check system status |

---

## Customization

### Change Default Model

Edit `app.py`:
```python
DEFAULT_MODEL = "gemma3:1b"  # Change to llama3.2:1b, etc.
```

Or use the `/model` command at runtime.

### Modify Personality

Edit the `SYSTEM_PROMPT` in `app.py` to change JetBrain's personality, knowledge, or communication style.

### Add New Slash Commands

Add new commands in the `handle_slash_command()` function in `app.py`.

---

## Troubleshooting

### "Ollama isn't running"
```bash
ollama serve
```

### "No module named flask"
```bash
source ../setup_env.sh
pip install flask
```

### Memory not persisting
Check that `.jetbrain` file exists and is writable:
```bash
ls -la .jetbrain
cat .jetbrain
```

### Slow responses
- Use a smaller model: `/model gemma3:1b`
- Close other GPU applications
- Check GPU: `tegrastats`

---

## Tech Stack

- **Backend**: Flask + Python 3.10
- **Database**: SQLite3 (chat history)
- **Memory**: JSON file (`.jetbrain`)
- **LLM**: Ollama
- **Vision**: MobileNetV2 (PyTorch + CUDA)
- **Frontend**: Vanilla JS + CSS

---

## Philosophy

JetBrain is designed to be more like Claude - a helpful assistant that remembers context and learns about you over time. The memory system allows it to build up knowledge about your preferences, projects, and working style.

Key principles:
- No emojis - clean, professional responses
- Persistent memory - learns and remembers
- Local first - everything runs on your Jetson
- Simple commands - easy to teach and manage

---

Built for Jetson by Lalo and JetBrain.
