from flask import Flask, render_template, request

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
model = AutoModelForCausalLM.from_pretrained("indolem/indobert-base-uncased")

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    if not msg.strip():  # Handling empty messages
        return "Bot: Maaf, saya tidak bisa merespons pesan kosong."
    return get_Chat_response(msg)

def get_Chat_response(text):
    try:
        chat_history_ids = torch.tensor([])  # Initialize chat history
        for step in range(5):  # Limit chat to 5 steps
            new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            text = response  # Set the next input as the current response
        return "Bot: " + response
    except Exception as e:  # Handle exceptions
        return "Bot: Maaf, terjadi kesalahan dalam mengolah pesan Anda."

if __name__ == '__main__':
    app.run(debug=True)
