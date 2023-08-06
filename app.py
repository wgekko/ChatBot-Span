from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



model = AutoModelForCausalLM.from_pretrained('ITG/DialoGPT-medium-spanish-chitchat')
tokenizer = AutoTokenizer.from_pretrained('ITG/DialoGPT-medium-spanish-chitchat')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

def get_Chat_response(text):  
    # Hablemos/chateamos  por 5 lineas
    for step in range(5):
        # codifique la nueva entrada del usuario, agregue el eos_token y devuelva un tensor en Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # agregar los nuevos tokens de entrada de usuario al historial de chat
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        #  generó una respuesta mientras limitaba el historial de chat total a 1000 tokens,
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # bonita impresión de los últimos tokens de salida del bot
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


if __name__ == '__main__':  
    app.run()
   