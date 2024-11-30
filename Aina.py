import os
import google.generativeai as genai
#from main import text_to_speech

from dotenv import load_dotenv
load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load system instruction from a file
def load_system_instruction(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Instruction file not found: {file_path}")
        return ""
    except Exception as e:
        print(f"Error reading instruction file: {str(e)}")
        return ""
    
system_instruction_file = "instruction.txt"
instruction = load_system_instruction(system_instruction_file)

# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
generation_config = {
  "temperature": 0,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
]

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro",
  safety_settings=safety_settings,
  generation_config=generation_config,
  system_instruction=instruction,
)



chat_session = model.start_chat(
    history=[]
)

print("Welcome to CellPay Customer Assistance!")
print()
#text_to_speech("Hello, how can I help you?")

while True:

    user_input = input("You: ")
    print()

    response = chat_session.send_message(user_input)

    model_response = response.text

    print(f'Limah: {model_response}')
    print()
    #text_to_speech(model_response)

    chat_session.history.append({"role": "user", "parts": [user_input]})
    chat_session.history.append({"role": "model", "parts": [model_response]})