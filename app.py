import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from streamlit_mic_recorder import mic_recorder
import yaml
import os
from utils import save_chat_history_json, load_chat_history_json, get_timestamp
from audio_handler import transcribe_audio
from llm_chain import load_normal_chain
from image_handler import handle_image

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

def load_chain(chat_history):
    return load_normal_chain(chat_history)

def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def track_index():
    st.session_state.session_index_tracker = st.session_state.session_key

def save_chat_history():
    if st.session_state.history != []:
        if st.session_state.session_key == "new_session":
            st.session_state.new_session_key = get_timestamp() + ".json"
            save_chat_history_json(st.session_state.history, config["chat_history_path"] + st.session_state.new_session_key)
        else:
            save_chat_history_json(st.session_state.history, config["chat_history_path"] + st.session_state.session_key)

def main():
    st.title("Multimodal Chat App")
    
    # Sidebar for chat sessions
    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + os.listdir(config["chat_history_path"])

    # Initialize session state variables
    if "send_input" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
    
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key is not None:
        st.session_state.session_key = st.session_state.new_session_key
        st.session_state.new_session_key = None

    # Chat session selection
    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select chat session", chat_sessions, key="session_key", index=index, on_change=track_index)

    # Load or initialize chat history
    if st.session_state.session_key != "new_session":
        st.session_state.history = load_chat_history_json(config["chat_history_path"] + st.session_state.session_key)
        print("Loaded history:", config["chat_history_path"] + st.session_state.session_key)
    else:
        st.session_state.history = []
    
    # Initialize chat history and LLM chain
    chat_history = StreamlitChatMessageHistory(key="history")
    llm_chain = load_chain(chat_history)

    # Create chat container for messages
    chat_container = st.container()

    # Display existing messages
    with chat_container:
        for message in chat_history.messages:
            st.chat_message(message.type).write(message.content)

    # Text input 
    user_input = st.text_input("Type your message here", key="user_input", on_change=set_send_input)

    # Voice recording and send button columns BELOW the text input
    voice_recording_column, send_button_column = st.columns(2)

    # Voice recording
    with voice_recording_column:
        voice_recording = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording", just_once=True)

    # Process voice recording
    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        print(transcribed_audio)
        llm_chain.run(transcribed_audio)

    # Send button
    with send_button_column:
        send_button = st.button("Send", key="send_button", on_click=clear_input_field)

    uploaded_audio=st.sidebar.file_uploader("Upload audio file", type=["wav", "mp3","ogg","m4a"])
    uploaded_image=st.sidebar.file_uploader("Upload image file", type=["jpg", "jpeg","png"])

    if uploaded_audio:
        transcribed_audio=transcribe_audio(uploaded_audio.getvalue())
        llm_chain.run("Summarize this text"+transcribed_audio)


    # Process text input
    if send_button or st.session_state.send_input:
        if uploaded_image:
            with st.spinner("Processing img..."):
                user_message="describe this image in detail please"
                if st.session_state.user_question != "":
                    user_message = st.session_state.user_question
                    st.session_state.user_question = ""
                llm_answer=handle_image(uploaded_image.getvalue(),st.session_state.user_question)
                chat_history.add_user_message(user_message)
                chat_history.add_ai_message(llm_answer)
        if st.session_state.user_question != "":
            llm_response = llm_chain.run(st.session_state.user_question)
            st.session_state.user_question = ""

    # Save chat history
    save_chat_history()

if __name__ == "__main__":
    main()