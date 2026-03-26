import streamlit as st
import requests
import os
import time
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional

# ===================================================================
# Page Configuration
# ===================================================================
st.set_page_config(
    page_title="ChatRAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================================================================
# Clean ChatGPT-Style CSS (No HTML in markdown)
# ===================================================================
st.markdown("""
<style>
    /* Hide all Streamlit elements */
    #MainMenu, footer, .stDeployButton, .stDecoration, header, .stToolbar {
        display: none !important;
    }
    
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for theming */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f7f7f8;
        --bg-tertiary: #ececec;
        --text-primary: #343541;
        --text-secondary: #565869;
        --border-light: #e5e5e5;
        --accent: #10a37f;
        --accent-hover: #0d8a69;
        --danger: #ef4444;
        --warning: #f59e0b;
        --success: #22c55e;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.1);
        --shadow: 0 2px 8px rgba(0,0,0,0.15);
        --radius: 8px;
        --radius-lg: 12px;
    }
    
    /* Dark theme support */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #343541;
            --bg-secondary: #444654;
            --bg-tertiary: #565869;
            --text-primary: #ececf1;
            --text-secondary: #c5c5d2;
            --border-light: #565869;
        }
    }
    
    /* Global app reset */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', system-ui, sans-serif !important;
        color: var(--text-primary) !important;
    }
    
    /* Main container - Full height layout */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
        height: 100vh !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* Top navigation bar */
    .top-nav {
        background: var(--bg-primary);
        border-bottom: 1px solid var(--border-light);
        padding: 0.75rem 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: sticky;
        top: 0;
        z-index: 100;
        box-shadow: var(--shadow-sm);
    }
    
    .nav-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    .nav-actions {
        display: flex;
        gap: 0.5rem;
        align-items: center;
    }
    
    /* Status indicator */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.375rem 0.75rem;
        font-size: 0.875rem;
        font-weight: 500;
        border-radius: 20px;
        border: 1px solid;
    }
    
    .status-online {
        background: rgba(34, 197, 94, 0.1);
        color: var(--success);
        border-color: var(--success);
    }
    
    .status-offline {
        background: rgba(239, 68, 68, 0.1);
        color: var(--danger);
        border-color: var(--danger);
    }
    
    /* Chat area container */
    .chat-area {
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    /* Messages container */
    .messages-container {
        flex: 1;
        overflow-y: auto;
        padding: 1rem 0;
        scroll-behavior: smooth;
    }
    
    /* Individual message */
    .chat-message {
        max-width: 768px;
        margin: 0 auto 1.5rem auto;
        padding: 0 1rem;
    }
    
    .message-row {
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
    }
    
    .message-avatar {
        width: 30px;
        height: 30px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.875rem;
        flex-shrink: 0;
        font-weight: 500;
    }
    
    .avatar-user {
        background: var(--accent);
        color: white;
    }
    
    .avatar-assistant {
        background: var(--bg-secondary);
        color: var(--text-primary);
        border: 1px solid var(--border-light);
    }
    
    .message-content {
        flex: 1;
        min-width: 0;
        color: var(--text-primary);
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    .message-content p {
        margin: 0 0 0.75rem 0;
    }
    
    .message-content p:last-child {
        margin-bottom: 0;
    }
    
    .message-content h1, .message-content h2, .message-content h3 {
        margin: 1rem 0 0.5rem 0;
        color: var(--text-primary);
    }
    
    .message-content ul, .message-content ol {
        margin: 0 0 0.75rem 0;
        padding-left: 1.25rem;
    }
    
    .message-content li {
        margin: 0.25rem 0;
    }
    
    .message-content code {
        background: var(--bg-tertiary);
        color: var(--text-primary);
        padding: 0.125rem 0.375rem;
        border-radius: 4px;
        font-family: 'SF Mono', Monaco, monospace;
        font-size: 0.875rem;
    }
    
    .message-content pre {
        background: var(--bg-secondary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius);
        padding: 1rem;
        overflow-x: auto;
        margin: 0.75rem 0;
    }
    
    /* Welcome screen */
    .welcome-screen {
        max-width: 768px;
        margin: 2rem auto;
        padding: 0 1rem;
        text-align: center;
    }
    
    .welcome-title {
        font-size: 2rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
    }
    
    .welcome-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
        line-height: 1.5;
    }
    
    /* Example buttons grid */
    .examples-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 0.75rem;
        margin-bottom: 2rem;
    }
    
    /* Input area - sticky at bottom */
    .input-area {
        border-top: 1px solid var(--border-light);
        background: var(--bg-primary);
        padding: 1rem;
        position: sticky;
        bottom: 0;
        z-index: 50;
        box-shadow: 0 -2px 8px rgba(0,0,0,0.05);
    }
    
    .input-container {
        max-width: 768px;
        margin: 0 auto;
        display: flex;
        gap: 0.75rem;
        align-items: flex-end;
    }
    
    /* Streamlit input overrides */
    .stTextArea {
        flex: 1;
        margin: 0;
    }
    
    .stTextArea > div > div > textarea {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: var(--radius-lg) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 0.875rem 1rem !important;
        resize: vertical !important;
        min-height: 44px !important;
        max-height: 120px !important;
        box-shadow: none !important;
        transition: border-color 0.2s ease !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent) !important;
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.1) !important;
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: var(--text-secondary) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius) !important;
        padding: 0.625rem 1.25rem !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        height: 44px !important;
        transition: background-color 0.2s ease !important;
        cursor: pointer !important;
    }
    
    .stButton > button:hover {
        background: var(--accent-hover) !important;
    }
    
    .stButton > button:disabled {
        background: var(--text-secondary) !important;
        opacity: 0.6 !important;
        cursor: not-allowed !important;
    }
    
    /* Metrics row */
    .metrics-row {
        display: flex;
        gap: 0.75rem;
        margin-top: 0.75rem;
        flex-wrap: wrap;
    }
    
    .metric-badge {
        background: var(--bg-secondary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius);
        padding: 0.375rem 0.75rem;
        font-size: 0.8125rem;
        color: var(--text-secondary);
        white-space: nowrap;
    }
    
    .metric-value {
        font-weight: 600;
        color: var(--text-primary);
    }
    
    /* Loading indicator */
    .loading-message {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-secondary);
        font-style: italic;
    }
    
    .typing-dots {
        display: flex;
        gap: 0.1875rem;
    }
    
    .typing-dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: var(--accent);
        animation: typing 1.4s ease-in-out infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { opacity: 0.3; transform: scale(0.8); }
        30% { opacity: 1; transform: scale(1); }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-light) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: var(--radius) !important;
        margin-top: 0.75rem !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-light) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius) var(--radius) !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .top-nav {
            padding: 0.5rem;
        }
        
        .nav-title {
            font-size: 1rem;
        }
        
        .welcome-title {
            font-size: 1.5rem;
        }
        
        .examples-grid {
            grid-template-columns: 1fr;
            gap: 0.5rem;
        }
        
        .input-area {
            padding: 0.75rem;
        }
        
        .input-container {
            gap: 0.5rem;
        }
        
        .metrics-row {
            gap: 0.5rem;
        }
    }
    
    /* Custom scrollbar */
    .messages-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .messages-container::-webkit-scrollbar-track {
        background: transparent;
    }
    
    .messages-container::-webkit-scrollbar-thumb {
        background: var(--border-light);
        border-radius: 3px;
    }
    
    .messages-container::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
</style>
""", unsafe_allow_html=True)

# ===================================================================
# Configuration
# ===================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_URL = f"{API_BASE_URL}/agentic_query"
HEALTH_URL = f"{API_BASE_URL}/health"

# ===================================================================
# Helper Functions
# ===================================================================
@st.cache_data(ttl=30)
def check_api_health():
    """Check API health with caching"""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        return "online" if response.status_code == 200 else "offline"
    except:
        return "offline"

def call_api(question: str, max_tokens: int = 1500, temperature: float = 0.1):
    """Call the API with error handling"""
    try:
        payload = {
            "question": question,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "include_plan": True,
            "top_k": 6
        }
        
        response = requests.post(API_URL, json=payload, timeout=120)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"Server error: {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to server"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    
def fix_markdown_formatting(text: str) -> str:
    """
    Fix markdown formatting issues to ensure proper rendering
    """
    import re
    
    # Ensure code blocks have proper spacing
    text = re.sub(r'```(\w+)\s*\n', r'```\1\n', text)
    text = re.sub(r'```\s*\n', r'```\n', text)
    
    # Fix broken code blocks (missing closing ```)
    code_block_count = text.count('```')
    if code_block_count % 2 != 0:
        # Odd number of ``` - add closing one
        text += '\n```'
    
    # Ensure paragraphs have proper spacing
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    
    # Fix list formatting
    text = re.sub(r'\n(\d+\.)', r'\n\n\1', text)  # Add space before numbered lists
    text = re.sub(r'\n(-|\*)', r'\n\n\1', text)  # Add space before bullet lists
    
    return text.strip()

def render_message(message: Dict, index: int):
    """Render a single message with proper formatting"""
    import re
    
    if message['type'] == 'user':
        # Escape HTML in user messages
        content = message['content'].replace('<', '&lt;').replace('>', '&gt;')
        
        st.markdown(f"""
        <div class="chat-message">
            <div class="message-row">
                <div class="message-avatar avatar-user">You</div>
                <div class="message-content">
                    <p>{content}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    elif message['type'] == 'assistant':
        # Use Streamlit's markdown for proper code rendering
        st.markdown(f"""
        <div class="chat-message">
            <div class="message-row">
                <div class="message-avatar avatar-assistant">AI</div>
                <div class="message-content">
        """, unsafe_allow_html=True)
        
        # Render content with Streamlit's markdown (supports code blocks properly)
        content = message['content']
        
        # Fix any remaining formatting issues
        content = fix_markdown_formatting(content)
        
        # Use st.markdown for proper rendering
        st.markdown(content)
        
        st.markdown("""
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        if 'metrics' in message:
            metrics = message['metrics']
            cached_text = " (cached)" if message.get('cached', False) else ""
            
            st.markdown(f"""
            <div class="chat-message">
                <div class="message-row">
                    <div class="message-avatar" style="opacity: 0;"></div>
                    <div class="metrics-row">
                        <div class="metric-badge">
                            <span class="metric-value">{metrics.get('confidence', 'N/A')}</span> confidence
                        </div>
                        <div class="metric-badge">
                            <span class="metric-value">{metrics.get('retrieved_docs', 0)}</span> sources
                        </div>
                        <div class="metric-badge">
                            <span class="metric-value">{metrics.get('processing_time', 0):.1f}s</span> processing{cached_text}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Expandable details
        if message.get('plan') or message.get('reasoning') or message.get('sources'):
            col1, col2 = st.columns([30, 738])
            
            with col2:
                if message.get('plan'):
                    with st.expander("🎯 Planning Process", expanded=False):
                        st.markdown(message['plan'])
                
                if message.get('reasoning'):
                    with st.expander("🧠 Reasoning Steps", expanded=False):
                        st.markdown(message['reasoning'])
                
                if message.get('sources'):
                    with st.expander(f"📚 Sources ({len(message['sources'])})", expanded=False):
                        for i, source in enumerate(message['sources'], 1):
                            st.write(f"{i}. {source}")
    
    elif message['type'] == 'error':
        st.markdown(f"""
        <div class="chat-message">
            <div class="message-row">
                <div class="message-avatar" style="background: var(--danger); color: white;">!</div>
                <div class="message-content" style="color: var(--danger);">
                    <p>{message['content']}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ===================================================================
# Session State
# ===================================================================
if 'conversations' not in st.session_state:
    st.session_state.conversations = [{"id": "default", "title": "New Chat", "messages": []}]
if 'current_conversation' not in st.session_state:
    st.session_state.current_conversation = "default"
if 'processing' not in st.session_state:
    st.session_state.processing = False

# ===================================================================
# Main Interface
# ===================================================================

# Top Navigation
api_status = check_api_health()
status_class = "status-online" if api_status == "online" else "status-offline"
status_icon = "●" if api_status == "online" else "●"
status_text = "Online" if api_status == "online" else "Offline"

st.markdown(f"""
<div class="top-nav">
    <h1 class="nav-title">ChatRAG</h1>
    <div class="nav-actions">
        <div class="status-badge {status_class}">
            <span style="color: {'var(--success)' if api_status == 'online' else 'var(--danger)'};">{status_icon}</span>
            <span>{status_text}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar for conversation management
with st.sidebar:
    st.markdown("### 💬 Conversations")
    
    # New Chat Button
    if st.button("+ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.conversations.append({
            "id": new_id,
            "title": "New Chat",
            "messages": []
        })
        st.session_state.current_conversation = new_id
        st.rerun()
    
    st.divider()
    
    # Conversation List
    for conv in st.session_state.conversations:
        is_current = conv["id"] == st.session_state.current_conversation
        title = conv["title"]
        
        # Truncate long titles
        if len(title) > 25:
            display_title = title[:22] + "..."
        else:
            display_title = title
        
        if st.button(
            display_title,
            key=f"conv_{conv['id']}",
            use_container_width=True,
            type="primary" if is_current else "secondary"
        ):
            st.session_state.current_conversation = conv["id"]
            st.rerun()
    
    st.divider()
    
    # Settings
    st.markdown("### ⚙️ Settings")
    max_tokens = st.slider("Response Length", 500, 3000, 1500, key="max_tokens")
    temperature = st.slider("Creativity", 0.0, 1.0, 0.1, key="temperature")
    
    st.divider()
    
    # Clear current chat
    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        for conv in st.session_state.conversations:
            if conv["id"] == st.session_state.current_conversation:
                conv["messages"] = []
                conv["title"] = "New Chat"
                break
        st.rerun()

# Get current conversation
current_conv = next(
    (conv for conv in st.session_state.conversations if conv["id"] == st.session_state.current_conversation),
    st.session_state.conversations[0]
)

# Chat Area
st.markdown('<div class="chat-area">', unsafe_allow_html=True)
st.markdown('<div class="messages-container">', unsafe_allow_html=True)

if not current_conv["messages"]:
    # Welcome Screen
    st.markdown("""
    <div class="welcome-screen">
        <h1 class="welcome-title">Welcome to ChatRAG</h1>
        <p class="welcome-subtitle">
            An intelligent document analysis assistant powered by advanced AI reasoning.
            Ask questions about your documents and get comprehensive answers with source citations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Example Questions as Streamlit Buttons
    st.markdown('<div class="examples-grid">', unsafe_allow_html=True)
    
    examples = [
        {"title": "🔍 Knowledge Lookup", "question": "Fetch insights from connected documents"},
        {"title": "🧩 Context Builder", "question": "Assemble context from multiple sources"},
        {"title": "🤝 Agent Collaboration", "question": "Coordinate agents for complex tasks"},
        {"title": "📊 Response Optimizer", "question": "Refine answers with feedback and scoring"}
    ]

    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(
                f"{example['title']}\n{example['question']}",
                key=f"example_{i}",
                use_container_width=True,
                help=f"Ask: {example['question']}"
            ):
                st.session_state.example_question = example['question']
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Display Messages
    for i, message in enumerate(current_conv["messages"]):
        render_message(message, i)

# Processing indicator
if st.session_state.processing:
    st.markdown("""
    <div class="chat-message">
        <div class="message-row">
            <div class="message-avatar avatar-assistant">AI</div>
            <div class="message-content">
                <div class="loading-message">
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                    <span>Thinking...</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close messages-container

# Input Area (Sticky at bottom)
st.markdown('<div class="input-area">', unsafe_allow_html=True)
st.markdown('<div class="input-container">', unsafe_allow_html=True)

# Handle example question
if hasattr(st.session_state, 'example_question'):
    user_input = st.session_state.example_question
    delattr(st.session_state, 'example_question')
    process_input = True
else:
    # Input form
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_area(
                label="Your Input",      # ✅ Only this, remove the ""
                placeholder="Message ChatRAG...",
                height=60,
                disabled=st.session_state.processing,
                key="message_input",
                label_visibility="collapsed"
            )
        
        with col2:
            process_input = st.form_submit_button(
                "Send",
                disabled=st.session_state.processing or api_status != "online",
                use_container_width=True
            )

st.markdown('</div>', unsafe_allow_html=True)  # Close input-container
st.markdown('</div>', unsafe_allow_html=True)  # Close input-area
st.markdown('</div>', unsafe_allow_html=True)  # Close chat-area

# ===================================================================
# Message Processing
# ===================================================================
if process_input and user_input and user_input.strip():
    if api_status != "online":
        st.error("System is offline. Please try again later.")
    else:
        # Add user message
        user_message = {
            'type': 'user',
            'content': user_input.strip(),
            'timestamp': datetime.now().isoformat()
        }
        
        current_conv["messages"].append(user_message)
        
        # Update conversation title if it's a new chat
        if current_conv["title"] == "New Chat" and len(current_conv["messages"]) == 1:
            # Use first 30 characters of the question as title
            title = user_input.strip()
            if len(title) > 30:
                title = title[:27] + "..."
            current_conv["title"] = title
        
        st.session_state.processing = True
        st.rerun()

# Handle processing
if st.session_state.processing and current_conv["messages"]:
    # Get the last user message
    user_messages = [msg for msg in current_conv["messages"] if msg['type'] == 'user']
    if user_messages:
        last_question = user_messages[-1]['content']
        
        # Call API
        start_time = time.time()
        result = call_api(last_question, max_tokens, temperature)
        processing_time = time.time() - start_time
        
        if result['success']:
            data = result['data']
            assistant_message = {
                'type': 'assistant',
                'content': data.get('answer', 'No response generated.'),
                'timestamp': datetime.now().isoformat(),
                'cached': data.get('cached', False),
                'sources': data.get('sources', []),
                'plan': data.get('plan', ''),
                'reasoning': data.get('reasoning', ''),
                'metrics': {
                    'confidence': data.get('confidence', 'Medium'),
                    'retrieved_docs': data.get('retrieved_docs', 0),
                    'processing_time': data.get('processing_time', processing_time)
                }
            }
        else:
            assistant_message = {
                'type': 'error',
                'content': f"Sorry, I encountered an error: {result['error']}",
                'timestamp': datetime.now().isoformat()
            }
        
        current_conv["messages"].append(assistant_message)
        st.session_state.processing = False
        st.rerun()

# Auto-scroll to bottom
if current_conv["messages"] or st.session_state.processing:
    st.markdown("""
    <script>
    setTimeout(() => {
        const container = document.querySelector('.messages-container');
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }, 100);
    </script>
    """, unsafe_allow_html=True)