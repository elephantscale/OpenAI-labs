# Building Your Own Database Agent

In **Building Your Own Database Agent lab**, we will develop an AI agent that interacts with databases using natural language, simplifying the process for querying and extracting insights.

- **Setup local environment to run lab 27-database-agent** 
     - Go inside 27-database-agent  folder
     - create venv using command
       - ```python3 -m venv 27```
     - Activate the new venv
        - ```source 27/bin/activate```
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
        ```
     - Start jupyter notebook
       - ```jupyter notebook```
     - Run any lab in jupyter notebook

## Course outline:

- **Focus areas**:
  - Focus on Retrieval-Augmented Generation (RAG) to build your first AI agent.
  - Deploy your Azure OpenAI Service instance.
  - Test the API.
  - Set up an orchestration engine like LangChain to enable these scenarios.

- **Lesson-1 and Lesson-2 focus on**:
  - Build an AI agent using langchain and Azure OpenAI
  - Load tabular data from a CSV file
  - Perform natural language queries using the Azure OpenAI Service to extract information.
  - Learn to reapply the agent to analyze your own CSV files.

- **Lesson-3 focus on implementing LangChain agents to connect to a provided SQL database**:
  - Build a DB agent that translates natural language to SQL code.

- **Lesson-4 focus on using Azure OpenAI Service’s function calling feature**:
  - Utilize pre-built functions for sending queries to databases.
  - Improve the efficiency and security of your SQL agent.

- **Lesson 5 focus on work with the Assistants API**:
  - Test it with the function calling and code interpreter features.
  - Enable connection to SQL databases and create your own DB agents more efficiently.


