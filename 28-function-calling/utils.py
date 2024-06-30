import inspect
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import requests


def raven_post(payload):
    """
    Sends a payload to a TGI endpoint.
    """
    # Now, let's prompt Raven!
    API_URL = "http://nexusraven.nexusflow.ai"
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def call_functioncalling_llm(prompt, api_to_call):
    """
    This function sends a request to the TGI endpoint to get Raven's function call.
    This will not generate Raven's justification and reasoning for the call, to save on latency.
    """
    signature = inspect.signature(api_to_call)
    docstring = api_to_call.__doc__
    prompt = f'''Function:\n{api_to_call.__name__}{signature}\n"""{clean_docstring(docstring)}"""\n\n\nUser Query:{prompt}<human_end>'''
    output = raven_post({
        "inputs": prompt,
        "parameters": {"temperature": 0.001, "stop": ["<bot_end>"], "do_sample": False, "max_new_tokens": 2048,
                       "return_full_text": False}})
    call = output[0]["generated_text"].replace("Call:", "").strip()
    return call


def query_raven(prompt):
    # This function sends a request to the TGI endpoint to get Raven's function call.
    # This will not generate Raven's justification and reasoning for the call, to save on latency.
    payload = {"inputs": prompt, "parameters": {"temperature": 0.001, "stop": ["<bot_end>"], "return_full_text": False,
                                                "do_sample": False, "max_new_tokens": 2048}}
    output = raven_post(payload)
    print(output)
    call = output[0]["generated_text"].replace("Call:", "").strip()
    return call

def clean_docstring(docstring):
    if docstring is not None:
        # Remove leading and trailing whitespace
        docstring = docstring.strip()
    return docstring


def draw_head(face_color='yellow', face_radius=0.4):
    """
    Creates the clown's head as a matplotlib patch.

    Parameters:
    - face_color (str): The color of the clown's face. Default is 'yellow'.
    - face_radius (float): The radius of the clown's face. Default is 0.4.

    Returns:
    - matplotlib.patches.Circle: A patch object representing the clown's head.
    """
    return patches.Circle((0.5, 0.5), face_radius, color=face_color, fill=True)


def draw_eyes(eye_color='black', eye_radius=0.05, eye_x_offset=0.15, eye_y_offset=0.15):
    """
    Creates the clown's eyes as matplotlib patches.

    Parameters:
    - eye_color (str): The color of the clown's eyes. Default is 'black'.
    - eye_radius (float): The radius of each eye. Default is 0.05.
    - eye_x_offset (float): The horizontal offset of each eye from the center. Default is 0.15.
    - eye_y_offset (float): The vertical offset of each eye from the center. Default is 0.15.

    Returns:
    - Tuple[matplotlib.patches.Circle, matplotlib.patches.Circle]: A tuple containing the patches for the left and right eyes.
    """
    eye_left = patches.Circle((0.5 - eye_x_offset, 0.5 + eye_y_offset), eye_radius, color=eye_color, fill=True)
    eye_right = patches.Circle((0.5 + eye_x_offset, 0.5 + eye_y_offset), eye_radius, color=eye_color, fill=True)
    return eye_left, eye_right


def draw_nose(nose_color='red', nose_radius=0.1, nose_x=0.5, nose_y=0.5):
    """
    Creates the clown's nose as a matplotlib patch.

    Parameters:
    - nose_color (str): The color of the clown's nose. Default is 'red'.
    - nose_radius (float): The radius of the nose. Default is 0.1.
    - nose_x (float): The x-coordinate of the nose's center. Default is 0.5.
    - nose_y (float): The y-coordinate of the nose's center. Default is 0.5.

    Returns:
    - matplotlib.patches.Circle: A patch object representing the clown's nose.
    """
    return patches.Circle((nose_x, nose_y), nose_radius, color=nose_color, fill=True)


def draw_mouth(mouth_color='black', mouth_width=0.3, mouth_height=0.1, mouth_x=0.5, mouth_y=0.3, mouth_theta1=200,
               mouth_theta2=340):
    """
    Creates the clown's mouth as a matplotlib patch.

    Parameters:
    - mouth_color (str): The color of the clown's mouth. Default is 'black'.
    - mouth_width (float): The width of the mouth arc. Default is 0.3.
    - mouth_height (float): The height of the mouth arc. Default is 0.1.
    - mouth_x (float): The x-coordinate of the mouth's center. Default is 0.5.
    - mouth_y (float): The y-coordinate of the mouth's center. Default is 0.3.
    - mouth_theta1 (float): The starting angle (in degrees) of the mouth arc. Default is 200.
    - mouth_theta2 (float): The ending angle (in degrees) of the mouth arc. Default is 340.

    Returns:
    - matplotlib.patches.Arc: A patch object representing the clown's mouth.
    """
    return patches.Arc((mouth_x, mouth_y), mouth_width, mouth_height, angle=0, theta1=mouth_theta1, theta2=mouth_theta2,
                       color=mouth_color, linewidth=2)


def draw_clown_face_parts(head, eyes, nose, mouth):
    """
    Draws a customizable clown face by assembling the provided components.

    Parameters:
    - head: The head patch returned by draw_head.
    - eyes: The tuple of eye patches returned by draw_eyes.
    - nose: The nose patch returned by draw_nose.
    - mouth: The mouth patch returned by draw_mouth.

    Example Use:
    draw_clown_face(head=draw_head("yellow", 0.4), eyes=draw_eyes("black", 0.05, 0.15, 0.15), nose=draw_nose("red", 0.1, 0.5, 0.5), mouth=draw_mouth("black", 0.3, 0.1, 0.5, 0.3, 200, 340))

    This function does not return a value but directly displays the assembled clown face using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.add_patch(head)
    ax.add_patch(eyes[0])
    ax.add_patch(eyes[1])
    ax.add_patch(nose)
    ax.add_patch(mouth)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()


def draw_clown_face(face_color='yellow', eye_color='black', nose_color='red',
                    eye_size=0.05, mouth_size=(0.3, 0.1), mouth_color='black',
                    eye_offset=(0.15, 0.15), mouth_theta=(200, 340)):
    """
    Draws a customizable, simplified clown face using matplotlib.

    Parameters:
    - face_color (str): Color of the clown's face. Default is 'yellow'.
    - eye_color (str): Color of the clown's eyes. Default is 'black'.
    - nose_color (str): Color of the clown's nose. Default is 'red'.
    - eye_size (float): Radius of the clown's eyes. Default is 0.05.
    - mouth_size (tuple): Width and height of the clown's mouth arc. Default is (0.3, 0.1).
    - eye_offset (tuple): Horizontal and vertical offset for the eyes from the center. Default is (0.15, 0.15).
    - mouth_theta (tuple): Starting and ending angles (in degrees) of the mouth arc. Default is (200, 340).

    This function creates a plot displaying a simplified clown face, where essential facial features' size, position, and color can be customized.

    Example usage:
    draw_clown_face(face_color='lightblue', eye_color='green', nose_color='orange',
                    eye_size=0.07, mouth_size=(0.4, 0.25),
                    eye_offset=(0.2, 0.2), mouth_theta=(0, 180))

    # This will draw a simplified clown face with a light blue face, green eyes, an orange nose, and a smiling mouth.
    """
    # Constants
    face_radius = 0.4
    nose_radius = 0.1
    nose_x, nose_y = 0.5, 0.5
    mouth_x, mouth_y = 0.5, 0.3

    fig, ax = plt.subplots(figsize=(2, 2))
    # Face
    face = patches.Circle((0.5, 0.5), face_radius, color=face_color, fill=True)
    ax.add_patch(face)
    # Eyes
    eye_left = patches.Circle((0.5 - eye_offset[0], 0.5 + eye_offset[1]), eye_size, color=eye_color, fill=True)
    eye_right = patches.Circle((0.5 + eye_offset[0], 0.5 + eye_offset[1]), eye_size, color=eye_color, fill=True)
    ax.add_patch(eye_left)
    ax.add_patch(eye_right)
    # Nose
    nose = patches.Circle((nose_x, nose_y), nose_radius, color=nose_color, fill=True)
    ax.add_patch(nose)
    # Mouth
    mouth = patches.Arc((mouth_x, mouth_y), mouth_size[0], mouth_size[1], angle=0,
                        theta1=mouth_theta[0], theta2=mouth_theta[1], color=mouth_color, linewidth=2)
    ax.add_patch(mouth)
    # Setting aspect ratio to 'equal' to ensure the face is circular
    ax.set_aspect('equal')
    # Remove axes
    ax.axis('off')
    plt.show()


def draw_tie():
    """
    Draws a tie
    """
    plt.figure(figsize=(1, 2))
    ax = plt.gca()

    # Top triangle loop
    top_triangle_vertices = [(0.25, 0.75), (0.5, 1), (0, 1)]
    top_triangle_loop = patches.Polygon(top_triangle_vertices, closed=True, color='black')
    ax.add_patch(top_triangle_loop)

    tail_triangle_vertices = [(0.25, 0.75), (0.5, 0), (0, 0)]
    tail_triangle_loop = patches.Polygon(tail_triangle_vertices, closed=True, color='black')
    ax.add_patch(tail_triangle_loop)

    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)

    ax.set_aspect('equal')
    plt.axis('off')
    plt.show()


def build_raven_prompt(function_list, user_query):
    import inspect
    raven_prompt = ""
    for function in function_list:
        signature = inspect.signature(function)
        docstring = function.__doc__
        prompt = \
            f'''
Function:
def {function.__name__}{signature}
    """
    {clean_docstring(docstring)}
    """
    
'''
        raven_prompt += prompt

    raven_prompt += f"User Query: {user_query}<human_end>"
    return raven_prompt


def create_random_database():
    import sqlite3
    import random

    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('toy_database.db')

    # Create a cursor object using the cursor() method
    cursor = conn.cursor()

    # Create table
    cursor.execute('''CREATE TABLE IF NOT EXISTS toys
                   (id INTEGER PRIMARY KEY, name TEXT, price REAL)''')

    # Define some random prefixes and suffixes for toy names
    prefixes = ['Magic', 'Super', 'Wonder', 'Mighty', 'Happy', 'Crazy']
    suffixes = ['Bear', 'Car', 'Doll', 'Train', 'Dragon', 'Robot']

    # Insert 100 sample data rows with random names
    for i in range(1, 101):
        toy_name = random.choice(prefixes) + ' ' + random.choice(suffixes)
        toy_price = round(random.uniform(5, 20), 2)  # Random price between 5 and 20
        cursor.execute('INSERT INTO toys (name, price) VALUES (?, ?)', (toy_name, toy_price))

    # Commit the transaction
    conn.commit()

    # Query the database
    cursor.execute('SELECT * FROM toys')
    print("Toys in database:")
    for row in cursor.fetchall():
        print(row)

    # Close the connection
    conn.close()


def query(payload):
    """
    Sends a payload to a TGI endpoint.
    """
    API_URL = "http://nexusraven.nexusflow.ai"
    headers = {
        "Content-Type": "application/json"
    }
    import requests
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def query_raven_with_reasoning(prompt):
    """
    This function sends a request to the TGI endpoint to get Raven's function call AND justification for the call
    """
    import requests
    output = query({
        "inputs": prompt,
        "parameters": {"temperature": 0.001, "do_sample": False, "max_new_tokens": 2000}})
    call = output[0]["generated_text"].replace("Call:", "").strip()
    return call


def execute_sql(sql_code: str):
    import sqlite3

    # Connect to the database
    conn = sqlite3.connect('toy_database.db')
    cursor = conn.cursor()

    cursor.execute('PRAGMA table_info(toys)')
    columns = [info[1] for info in cursor.fetchall()]  # Extracting the column names

    # Query to select all data
    cursor.execute(sql_code)
    rows = cursor.fetchall()

    return_string = " ".join(columns)
    for idx, row in enumerate(rows):
        row = (idx, *row)
        return_string += "\n" + str(row)

    # Close the connection
    conn.close()
    return return_string
