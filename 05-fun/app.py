#!/usr/bin/env python3
"""
JetBrain - A smart AI chatbot for Jetson
Flask app with Ollama LLM, vision analysis, persistent memory, and chat history
"""

import os
import re
import sqlite3
import uuid
import json
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
DATABASE = os.path.join(os.path.dirname(__file__), 'jetbrain.db')
MEMORY_FILE = os.path.join(os.path.dirname(__file__), '.jetbrain')
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:1b"

# JetBrain's personality - more natural, no emojis
SYSTEM_PROMPT = """You are JetBrain, an AI assistant running locally on a Jetson Orin Nano.

About your user:
- Their name is Lalo
- You work together on coding projects
- They use Linux, Mac, and Windows machines
- They enjoy building things and learning

Your communication style:
- Be helpful, direct, and conversational
- No emojis in your responses
- Be natural - you can be witty or casual when appropriate
- You're knowledgeable about AI, edge computing, CUDA, and software development
- You remember things Lalo tells you (check your memory section)
- Be genuine - if you don't know something, say so

IMPORTANT - Code responses:
- When providing code, ALWAYS provide the complete file
- Never split code across multiple responses
- Never say "here's the first part" or "continued in next message"
- Code should be complete, working, and in a single code block
- Include all imports, functions, and necessary components
- If a file is long, that's fine - provide it in its entirety

You have access to persistent memory. When Lalo tells you to remember something, it gets saved and you'll have access to it in future conversations. Use this memory to be more helpful and contextual.

Do not use emojis. Ever. Keep responses clean and professional but friendly."""

# Vision model (lazy loaded)
vision_model = None
imagenet_labels = None

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============== Memory System ==============

def load_memory():
    """Load persistent memory from .jetbrain file"""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return {"facts": [], "preferences": [], "projects": [], "notes": []}
    return {"facts": [], "preferences": [], "projects": [], "notes": []}


def save_memory(memory):
    """Save memory to .jetbrain file"""
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)


def add_to_memory(category, item):
    """Add an item to memory"""
    memory = load_memory()
    if category not in memory:
        memory[category] = []

    # Avoid duplicates
    if item not in memory[category]:
        memory[category].append(item)
        save_memory(memory)
        return True
    return False


def remove_from_memory(category, item):
    """Remove an item from memory"""
    memory = load_memory()
    if category in memory and item in memory[category]:
        memory[category].remove(item)
        save_memory(memory)
        return True
    return False


def clear_memory_category(category):
    """Clear a category from memory"""
    memory = load_memory()
    if category in memory:
        memory[category] = []
        save_memory(memory)
        return True
    return False


def format_memory_for_context():
    """Format memory as context for the LLM"""
    memory = load_memory()

    sections = []

    if memory.get("facts"):
        sections.append("Facts about Lalo:\n" + "\n".join(f"- {fact}" for fact in memory["facts"]))

    if memory.get("preferences"):
        sections.append("Preferences:\n" + "\n".join(f"- {pref}" for pref in memory["preferences"]))

    if memory.get("projects"):
        sections.append("Current projects:\n" + "\n".join(f"- {proj}" for proj in memory["projects"]))

    if memory.get("notes"):
        sections.append("Notes:\n" + "\n".join(f"- {note}" for note in memory["notes"]))

    if sections:
        return "\n\n[MEMORY]\n" + "\n\n".join(sections) + "\n[/MEMORY]\n"
    return ""


def handle_slash_command(message):
    """
    Handle slash commands
    Returns (response, should_continue) tuple
    """
    global DEFAULT_MODEL
    message = message.strip()

    # /help - show available commands
    if message == "/help":
        return """Available commands:

/remember <text> - Save something to memory
/forget <text> - Remove something from memory
/memory - Show all saved memories
/clear <category> - Clear a memory category (facts, preferences, projects, notes)
/status - Check system status
/model <name> - Switch LLM model (e.g., /model llama3.2:1b)

Memory categories:
- facts: Things about you (e.g., "/remember fact: I work at NVIDIA")
- preferences: Your preferences (e.g., "/remember preference: I prefer Python over JavaScript")
- projects: Current projects (e.g., "/remember project: Building a robot arm")
- notes: General notes (e.g., "/remember note: Meeting on Friday")

Example: /remember fact: My favorite language is Python""", False

    # /memory - show current memory
    if message == "/memory":
        memory = load_memory()
        if not any(memory.values()):
            return "Memory is empty. Use /remember to add things.", False

        output = "Current memory:\n\n"
        for category, items in memory.items():
            if items:
                output += f"{category.upper()}:\n"
                for item in items:
                    output += f"  - {item}\n"
                output += "\n"
        return output.strip(), False

    # /remember - add to memory
    if message.startswith("/remember "):
        content = message[10:].strip()

        # Parse category if specified
        category = "notes"  # default
        if ":" in content:
            prefix, text = content.split(":", 1)
            prefix = prefix.lower().strip()
            if prefix in ["fact", "facts"]:
                category = "facts"
                content = text.strip()
            elif prefix in ["preference", "preferences", "pref"]:
                category = "preferences"
                content = text.strip()
            elif prefix in ["project", "projects"]:
                category = "projects"
                content = text.strip()
            elif prefix in ["note", "notes"]:
                category = "notes"
                content = text.strip()

        if add_to_memory(category, content):
            return f"Got it. I'll remember that. (saved to {category})", False
        else:
            return "I already have that in my memory.", False

    # /forget - remove from memory
    if message.startswith("/forget "):
        content = message[8:].strip()
        memory = load_memory()

        # Try to find and remove from any category
        removed = False
        for category in memory:
            if content in memory[category]:
                remove_from_memory(category, content)
                removed = True
                break

        if removed:
            return f"Removed from memory: {content}", False
        else:
            return "I couldn't find that in my memory.", False

    # /clear - clear a category
    if message.startswith("/clear "):
        category = message[7:].strip().lower()
        if category in ["facts", "preferences", "projects", "notes"]:
            clear_memory_category(category)
            return f"Cleared all {category} from memory.", False
        elif category == "all":
            save_memory({"facts": [], "preferences": [], "projects": [], "notes": []})
            return "Cleared all memory.", False
        else:
            return f"Unknown category: {category}. Use: facts, preferences, projects, notes, or all", False

    # /status - system status
    if message == "/status":
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else "N/A"

        ollama_ok = False
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            ollama_ok = r.status_code == 200
        except:
            pass

        memory = load_memory()
        mem_count = sum(len(v) for v in memory.values())

        return f"""System Status:

CUDA: {'Available' if cuda else 'Not available'}
GPU: {gpu}
Ollama: {'Running' if ollama_ok else 'Not running'}
Memory items: {mem_count}
Model: {DEFAULT_MODEL}""", False

    # /model - switch model
    if message.startswith("/model "):
        new_model = message[7:].strip()
        DEFAULT_MODEL = new_model
        return f"Switched to model: {new_model}", False

    # Not a command
    return None, True


# ============== Database ==============

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database"""
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            image_path TEXT,
            image_analysis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );
    ''')
    conn.commit()
    conn.close()


# ============== Vision Model ==============

def load_vision_model():
    """Load MobileNetV2 for image analysis"""
    global vision_model, imagenet_labels

    if vision_model is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vision_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        vision_model = vision_model.to(device)
        vision_model.eval()

        # Load labels
        labels_file = os.path.join(os.path.dirname(__file__), '..', '04-vision-llm-integration', 'imagenet_labels.txt')
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                imagenet_labels = [line.strip() for line in f.readlines()]
        else:
            try:
                url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
                response = requests.get(url, timeout=10)
                imagenet_labels = response.text.strip().split('\n')
            except:
                imagenet_labels = [f"class_{i}" for i in range(1000)]

    return vision_model, imagenet_labels


def analyze_image(image_path):
    """Analyze an image with MobileNetV2"""
    model, labels = load_vision_model()
    device = next(model.parameters()).device

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output[0], dim=0)

    top5_prob, top5_idx = torch.topk(probs, 5)

    results = []
    for prob, idx in zip(top5_prob, top5_idx):
        results.append({
            'label': labels[idx.item()],
            'confidence': f"{prob.item()*100:.1f}%"
        })

    return results


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============== LLM ==============

def query_ollama(prompt, model=None, context=""):
    """Query Ollama LLM"""
    if model is None:
        model = DEFAULT_MODEL

    # Include memory in context
    memory_context = format_memory_for_context()

    full_prompt = f"{SYSTEM_PROMPT}\n\n{memory_context}{context}\n\nUser: {prompt}\n\nJetBrain:"

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False
            },
            timeout=120
        )

        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: Ollama returned status {response.status_code}"

    except requests.exceptions.ConnectionError:
        return "Ollama isn't running. Start it with: ollama serve"
    except Exception as e:
        return f"Error: {str(e)}"


# ============== Routes ==============

@app.route('/')
def index():
    """Render the main chat interface"""
    if 'conversation_id' not in session:
        session['conversation_id'] = None
    return render_template('index.html')


@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get all conversations"""
    conn = get_db()
    conversations = conn.execute(
        'SELECT * FROM conversations ORDER BY updated_at DESC'
    ).fetchall()
    conn.close()

    return jsonify([dict(c) for c in conversations])


@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    """Create a new conversation"""
    conv_id = str(uuid.uuid4())
    title = request.json.get('title', 'New Chat')

    conn = get_db()
    conn.execute(
        'INSERT INTO conversations (id, title) VALUES (?, ?)',
        (conv_id, title)
    )
    conn.commit()
    conn.close()

    session['conversation_id'] = conv_id
    return jsonify({'id': conv_id, 'title': title})


@app.route('/api/conversations/<conv_id>', methods=['DELETE'])
def delete_conversation(conv_id):
    """Delete a conversation"""
    conn = get_db()
    conn.execute('DELETE FROM messages WHERE conversation_id = ?', (conv_id,))
    conn.execute('DELETE FROM conversations WHERE id = ?', (conv_id,))
    conn.commit()
    conn.close()

    return jsonify({'success': True})


@app.route('/api/conversations/<conv_id>/messages', methods=['GET'])
def get_messages(conv_id):
    """Get messages for a conversation"""
    conn = get_db()
    messages = conn.execute(
        'SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at',
        (conv_id,)
    ).fetchall()
    conn.close()

    return jsonify([dict(m) for m in messages])


@app.route('/api/chat', methods=['POST'])
def chat():
    """Send a message and get a response"""
    conv_id = request.form.get('conversation_id')
    message = request.form.get('message', '')
    image = request.files.get('image')

    # Check for slash commands first
    if message.startswith('/'):
        cmd_response, should_continue = handle_slash_command(message)
        if not should_continue:
            return jsonify({
                'conversation_id': conv_id,
                'response': cmd_response,
                'is_command': True
            })

    # Create conversation if needed
    if not conv_id:
        conv_id = str(uuid.uuid4())
        title = message[:50] + '...' if len(message) > 50 else message
        if not title:
            title = "Image Chat"

        conn = get_db()
        conn.execute(
            'INSERT INTO conversations (id, title) VALUES (?, ?)',
            (conv_id, title)
        )
        conn.commit()
        conn.close()

    image_path = None
    image_analysis = None
    context = ""

    # Handle image upload
    if image and allowed_file(image.filename):
        filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        # Analyze image
        try:
            analysis = analyze_image(image_path)
            image_analysis = json.dumps(analysis)

            # Add to context
            top_predictions = ", ".join([f"{a['label']} ({a['confidence']})" for a in analysis[:3]])
            context = f"\n[Image uploaded. Vision analysis: {top_predictions}]\n"
        except Exception as e:
            context = f"\n[Image uploaded but analysis failed: {str(e)}]\n"

    # Get conversation history for context
    conn = get_db()
    recent_messages = conn.execute(
        '''SELECT role, content FROM messages
           WHERE conversation_id = ?
           ORDER BY created_at DESC LIMIT 10''',
        (conv_id,)
    ).fetchall()
    conn.close()

    # Build conversation context
    history = ""
    for msg in reversed(recent_messages):
        role = "User" if msg['role'] == 'user' else "JetBrain"
        history += f"{role}: {msg['content'][:500]}\n"

    full_context = history + context

    # Save user message
    conn = get_db()
    conn.execute(
        '''INSERT INTO messages (conversation_id, role, content, image_path, image_analysis)
           VALUES (?, ?, ?, ?, ?)''',
        (conv_id, 'user', message, image_path, image_analysis)
    )
    conn.commit()

    # Get response from Ollama
    if message or image_analysis:
        prompt = message if message else "What do you see in this image?"
        if image_analysis and message:
            prompt = f"[Looking at the uploaded image] {message}"

        response = query_ollama(prompt, context=full_context)
    else:
        response = "Send me a message or an image and let's chat."

    # Save assistant response
    conn.execute(
        'INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)',
        (conv_id, 'assistant', response)
    )

    # Update conversation timestamp
    conn.execute(
        'UPDATE conversations SET updated_at = ? WHERE id = ?',
        (datetime.now(), conv_id)
    )
    conn.commit()
    conn.close()

    return jsonify({
        'conversation_id': conv_id,
        'response': response,
        'image_analysis': json.loads(image_analysis) if image_analysis else None
    })


@app.route('/api/memory', methods=['GET'])
def get_memory():
    """Get current memory"""
    return jsonify(load_memory())


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            return jsonify({'models': models, 'current': DEFAULT_MODEL})
        return jsonify({'models': [], 'current': DEFAULT_MODEL, 'error': 'Failed to fetch'})
    except Exception as e:
        return jsonify({'models': [], 'current': DEFAULT_MODEL, 'error': str(e)})


@app.route('/api/model', methods=['POST'])
def set_model():
    """Set the current model"""
    global DEFAULT_MODEL
    new_model = request.json.get('model')
    if new_model:
        DEFAULT_MODEL = new_model
        return jsonify({'success': True, 'model': DEFAULT_MODEL})
    return jsonify({'success': False, 'error': 'No model specified'})


@app.route('/api/status', methods=['GET'])
def status():
    """Check system status"""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None

    # Check Ollama
    ollama_status = False
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        ollama_status = r.status_code == 200
    except:
        pass

    memory = load_memory()
    memory_count = sum(len(v) for v in memory.values())

    return jsonify({
        'cuda': cuda_available,
        'gpu': gpu_name,
        'ollama': ollama_status,
        'memory_count': memory_count,
        'model': DEFAULT_MODEL
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    init_db()
    print("=" * 60)
    print("JetBrain - Your Jetson AI Companion")
    print("=" * 60)
    print(f"CUDA: {'Available' if torch.cuda.is_available() else 'Not available'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {DEFAULT_MODEL}")

    memory = load_memory()
    mem_count = sum(len(v) for v in memory.values())
    print(f"Memory items: {mem_count}")

    print("\nStarting server at http://localhost:5000")
    print("Type /help in chat for available commands")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
