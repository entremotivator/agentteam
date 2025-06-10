import streamlit as st
from openai import OpenAI
import tempfile
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

st.set_page_config(page_title="Team AI Chatbot", layout="wide")

# Default team member personas
default_team = {
    "Alex (Strategist)": "You are Alex, a strategic planner focused on long-term goals and market insights.",
    "Becky (Marketing Guru)": "You are Becky, a creative marketing expert who knows how to grow a brand.",
    "Chris (Tech Lead)": "You are Chris, a technical genius who explains complex systems simply.",
    "Dana (Sales Pro)": "You are Dana, a persuasive and confident sales expert focused on conversions.",
    "Eli (Data Analyst)": "You are Eli, a sharp data analyst who spots trends and patterns easily."
}

# Initialize team in session state
if "team_prompts" not in st.session_state:
    st.session_state.team_prompts = default_team.copy()

# Sidebar — system prompt
st.sidebar.title("🧠 AI Persona & Team Settings")

# Editable team prompts
st.sidebar.subheader("✏️ Edit Team Member Personas")
for name in list(st.session_state.team_prompts.keys()):
    updated_prompt = st.sidebar.text_area(
        f"{name}", st.session_state.team_prompts[name], key=f"prompt_{name}"
    )
    st.session_state.team_prompts[name] = updated_prompt

# Choose active member
if "selected_member" not in st.session_state:
    st.session_state.selected_member = list(st.session_state.team_prompts.keys())[0]

selected_member = st.sidebar.selectbox(
    "🧑‍💼 Choose Active Team Member", list(st.session_state.team_prompts.keys()),
    index=list(st.session_state.team_prompts.keys()).index(st.session_state.selected_member)
)

# --- Document Upload Section ---
st.sidebar.subheader("📄 Import Knowledge Base (CSV, TXT, PDF)")
uploaded_file = st.sidebar.file_uploader(
    "Upload a document", type=["csv", "txt", "pdf"]
)

if uploaded_file:
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Load document based on type
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif uploaded_file.name.endswith(".csv"):
        loader = CSVLoader(file_path)
    else:
        loader = TextLoader(file_path)

    # Load and split into manageable chunks
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_chunks = splitter.split_documents(docs)

    # Combine all chunks into a single string (for context)
    document_text = "\n\n".join([chunk.page_content for chunk in doc_chunks])

    # Store in session state for use in chat
    st.session_state["knowledge_base"] = document_text

    st.sidebar.success("Document imported! Your AI can now reference it.")
else:
    st.session_state["knowledge_base"] = ""

# --- End Document Upload Section ---

# Detect change and update system prompt
if "last_selected" not in st.session_state:
    st.session_state.last_selected = selected_member

# System prompt includes editable persona and knowledge base
if selected_member != st.session_state.last_selected or "messages" not in st.session_state:
    st.session_state.last_selected = selected_member

    # Add knowledge base to the system prompt if available
    kb = st.session_state.get("knowledge_base", "")
    persona = st.session_state.team_prompts[selected_member]
    if kb:
        system_prompt = (
            f"{persona}\n\n"
            f"You also have access to the following document for reference:\n"
            f"{kb[:3000]}..."  # Limit to 3000 chars for token safety
        )
    else:
        system_prompt = persona

    st.session_state.messages = [{
        "role": "system",
        "content": system_prompt
    }]

# Chat UI
st.title("🤖 Team AI Chatbot")
st.markdown(f"**Chatting with:** {selected_member}")

if st.session_state.get("knowledge_base"):
    st.info("Knowledge base loaded. The AI will use it to answer your questions.")

# Display message history
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=st.session_state.messages
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"⚠️ Error: {e}"
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

