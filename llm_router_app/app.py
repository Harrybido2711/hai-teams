import gradio as gr
from utils.transcriber import transcribe_file
from utils.query_classifier import classify_query
from router import route_query

chat_history = []

def process_input(user_input, file_upload):
    if file_upload:
        transcription = transcribe_file(file_upload)
        user_input = transcription

    classification = classify_query(user_input)
    model_name, response = route_query(user_input, classification)
    chat_history.append((user_input, f"[{model_name}] {response}"))
    return chat_history

with gr.Blocks() as demo:
    gr.Markdown("# 🔀 Multi-LLM Router Chat")
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column():
            txt_input = gr.Textbox(placeholder="Ask me anything...", label="Your query")
            file_input = gr.File(label="Optional audio/video upload")
        with gr.Column():
            submit_btn = gr.Button("Submit")
    
    submit_btn.click(process_input, inputs=[txt_input, file_input], outputs=chatbot)

demo.launch()