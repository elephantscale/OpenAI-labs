# Function-Calling and Data Extraction with NexusRavenV2-13B:

- **Setup local environment to run lab 28-database-agent** 
     - Go inside 28-function-calling folder
     - create venv using command
       - ```python3 -m venv 28```
     - Activate the new venv
        - ```source 28/bin/activate```
     - Next install jupyter notebook in the same venv
        - ```pip3 install jupyter```
     - Install lab requirements
       - ``pip3 install -r requirements.txt``
     - Create one .env file for the keys and add the values
        ```
        OPENAI_API_VERSION=
        AZURE_DEPLOYMENT=
        OPENAI_API_KEY=
        AZURE_ENDPOINT=
        TAVILY_API_KEY=
        ```
     - Start jupyter notebook
       - ```jupyter notebook```
     - Run any lab in jupyter notebook
- Overview

In this lab is specific to below two points:
- Extend LLMs with custom functionality via function-calling, enabling them to form calls to external functions.
- Extract structured data from natural language inputs, making real-world data usable for analysis

## Lesson-1:
- Function-Calling: Integrate function-calling in detail. Construct prompts with function definitions and utilize LLM responses to execute these functions.
## Lesson-2:
- Complex Function Calls: LLMs are capable of handling multiple function calls, including parallel and nested calls. This capability enables creating intricate agent workflows where an LLM plans and executes a sequence of function calls to accomplish tasks.
## Lesson-3:
- Using OpenAPI specifications construct function calls that interact with web services. This allows LLMs to access and utilize data from external APIs seamlessly.
## Lesson-4
- Using function-call extract structured data from natural language inputs. This feature enhances the ability to parse and utilize information contained within unstructured textual data.
## Lesson-5:
- Building applications that automate tasks such as generating SQL queries from customer service transcripts. LLM-generated commands can be used to execute database operations efficiently.

