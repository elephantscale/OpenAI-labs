# Building Your Own Database Agent

In the **Building Your Own Database Agent lab**, we will develop an AI agent that interacts with databases using natural language, simplifying the process for querying and extracting insights.

* Documentation
  * Explain the virtual environment
  * Virtual environment 27

* Course in DeeplearningAI
* https://learn.deeplearning.ai/courses/building-your-own-database-agent/lesson/1/introduction

* 
## Setup Local Environment for Lab 27 - Database Agent

1. **Navigate to the Project Directory**
   - Go inside the `27-database-agent` folder.

2. **Create a Virtual Environment**
   - Run the following command to create a virtual environment:
     ```sh
     python3 -m venv 27
     ```

3. **Activate the Virtual Environment**
   - Activate the new virtual environment using:
     ```sh
     source 27/bin/activate
     ```

4. **Install Jupyter Notebook**
   - Install Jupyter Notebook in the same virtual environment:
     ```sh
     pip3 install jupyter
     ```

5. **Install IPython Kernel**
   - Install `ipykernel` to attach the Jupyter environment to the same kernel:
     ```sh
     pip3 install ipykernel
     ```

6. **Add Environment to Jupyter Kernel**
   - Add the current environment to the Jupyter kernel:
     ```sh
     python3 -m ipykernel install --user --name=27 --display-name "Python (lab 27)"
     ```

7. **Install Lab Requirements**
   - Install the lab requirements specified in the `requirements.txt` file:
     ```sh
     pip3 install -r requirements.txt
     ```

8. **Create .env File for API Keys**
   - Create a `.env` file for the keys and add the following values:
     ```sh
     OPENAI_API_VERSION=
     AZURE_DEPLOYMENT=
     OPENAI_API_KEY=
     AZURE_ENDPOINT=
     ```

9. **Start Jupyter Notebook**
   - Start Jupyter Notebook with the command:
     ```sh
     jupyter notebook
     ```

10. **Select the Correct Kernel**
    - At the top right corner of the Jupyter Notebook, click on the kernel and choose "Python (lab 27)".

11. **Verify the Environment**
    - To verify that you are in the correct environment, create a new cell and run the following command. If the output shows your current environment, you are all set:
      ```python
      import sys
      print(sys.executable)
      ```

12. **Run the Labs**
    - You are now ready to run any lab in Jupyter Notebook.

## Course Outline

### Focus Areas
- Focus on Retrieval-Augmented Generation (RAG) to build your first AI agent.
- Deploy your Azure OpenAI Service instance.
- Test the API.
- Set up an orchestration engine like LangChain to enable these scenarios.

### Lesson 1 and Lesson 2
- Build an AI agent using LangChain and Azure OpenAI.
- Load tabular data from a CSV file.
- Perform natural language queries using the Azure OpenAI Service to extract information.
- Learn to reapply the agent to analyze your own CSV files.

### Lesson 3
- Implement LangChain agents to connect to a provided SQL database.
- Build a DB agent that translates natural language to SQL code.

### Lesson 4
- Use Azure OpenAI Service’s function calling feature.
- Utilize pre-built functions for sending queries to databases.
- Improve the efficiency and security of your SQL agent.

### Lesson 5
- Work with the Assistants API.
- Test it with the function calling and code interpreter features.
- Enable connection to SQL databases and create your own DB agents more efficiently.


### Lesson 6
- Build A dialogue feature extraction pipeline using function calling!
