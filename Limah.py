import os
import google.generativeai as genai
from dotenv import load_dotenv
import tkinter as tk
from tkinter import scrolledtext, messagebox

# Load environment variables
load_dotenv()

# Configure the generative model
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load system instruction
def load_system_instruction(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "Instruction file not found."
    except Exception as e:
        return f"Error reading instruction file: {str(e)}"

system_instruction_file = "instruction.txt"
instruction = load_system_instruction(system_instruction_file)

generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction=instruction,
)

chat_session = model.start_chat(history=[])

# Define the Tkinter app
class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Limah - The CellPay Virtual Assistant")
        self.root.geometry("600x700")

        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state="disabled", height=30, width=70)
        self.chat_display.pack(pady=10)

        # User input field
        self.input_field = tk.Entry(root, width=70)
        self.input_field.pack(pady=10)
        self.input_field.bind("<Return>", self.send_message)

        # Send button
        self.send_button = tk.Button(root, text="Send", command=self.send_message)
        self.send_button.pack(pady=5)

        # Initialize conversation
        self.display_message("Limah", "Welcome to CellPay Customer Assistance!")

    def display_message(self, sender, message):
        """Display a message in the chat display area."""
        self.chat_display.config(state="normal")
        self.chat_display.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state="disabled")

    def send_message(self, event=None):
        """Handle sending a message and getting a response."""
        user_message = self.input_field.get().strip()
        if not user_message:
            messagebox.showwarning("Warning", "Please enter a message!")
            return

        # Display user message
        self.display_message("You", user_message)
        self.input_field.delete(0, tk.END)

        try:
            # Get response from the model
            response = chat_session.send_message(user_message)
            model_response = response.text
            self.display_message("Limah", model_response)

            # Update session history
            chat_session.history.append({"role": "user", "parts": [user_message]})
            chat_session.history.append({"role": "model", "parts": [model_response]})

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
