import gradio as gr
import json, os, time

chat_file = "chat.json"

shared_chat_history = []

def chat_fn(message, history):
    global shared_chat_history
    response = f"Echo: {message}"
    shared_chat_history.append({"role": "user", "content": message})
    shared_chat_history.append({"role": "assistant", "content": response})
    save_history()
    return response

def save_history():
    with open(chat_file, 'w') as f:
        json.dump(shared_chat_history, f)

def poll_chat():
    return shared_chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type='messages')

    interface = gr.ChatInterface(fn=chat_fn, type='messages', title='Unimodal Router', chatbot=chatbot)

    timer = gr.Timer(2)
    timer.tick(poll_chat, outputs=chatbot)

demo.launch(share=True)