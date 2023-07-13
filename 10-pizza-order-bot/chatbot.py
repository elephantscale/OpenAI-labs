import os
import openai
import panel as pn
from dotenv import load_dotenv, find_dotenv
from panel.io.server import get_server
from panel.template import BootstrapTemplate

# Load the environment variables
load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')  # Get the OpenAI API key


# This function generates a response from the chat model
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message["content"]


# This function generates a response based on a list of messages
def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
    return response.choices[0].message["content"]


# This function collects messages and generates a response
def collect_messages(_):
    prompt = inp.value
    inp.value = ''
    context.append({'role': 'user', 'content': prompt})
    response = get_completion_from_messages(context)
    context.append({'role': 'assistant', 'content': response})
    panels.append(pn.Row('User:', pn.pane.Markdown(prompt)))
    panels.append(pn.Row('Assistant:', pn.pane.Markdown(response)))
    return pn.Column(*panels)


# Initialize the GUI components
template = BootstrapTemplate(title='Order Pizza Bot')

# Load the context message from an external file
with open('context.txt', 'r') as file:
    system_message = file.read().replace('\n', '')


panels = []  # Stores the panel Rows
context = [{'role': 'system', 'content': system_message}]  # Initial context
inp = pn.widgets.TextInput(value="Hi", placeholder='Enter text here…')  # Input text field
button_conversation = pn.widgets.Button(name="Chat!")  # Chat button
interactive_conversation = pn.bind(collect_messages, button_conversation)  # Interactive conversation widget

# Create a chat box which includes the text input and the chat button
chat_box = pn.Row(inp, button_conversation)


# Define quick response buttons and their actions
quick_responses = ["Hello", "Order pizza", "Order drink", "Delivery", "End chat"]

def quick_response_click(event):
    inp.value = event.obj.name  # Set the input value to the name of the button
    button_conversation.clicks += 1  # Simulate click on the chat button
    #print("A")
    #collect_messages(None)  # Collect messages

quick_response_buttons = [pn.widgets.Button(name=response, width=150) for response in quick_responses]
for button in quick_response_buttons:
    button.on_click(quick_response_click)

# Create a row with quick response buttons
quick_response_row = pn.Row(*quick_response_buttons, css_classes=["container"], margin=(10, 0, 10, 0))



# Create the dashboard which includes the chat box, quick responses and the interactive conversation widget
dashboard = pn.Column(
    pn.panel(interactive_conversation, loading_indicator=True),
    chat_box,
    quick_response_row,
    css_classes=["container"],  # Add 'container' CSS class
)

template.main.append(dashboard)  # Append the dashboard to the template

# Serve the application
server = get_server(template, port=5000)  # Initialize the server
server.start()  # Start the server
server.io_loop.start()  # Start the I/O loop
