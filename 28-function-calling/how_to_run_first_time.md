## Setup Local Environment for Lab 28 - Database Agent

1. **Navigate to the Project Directory**
   - Go inside the `28-function-calling` folder.

2. **Create a Virtual Environment**
   - Run the following command to create a virtual environment:
     ```sh
     python3 -m venv 28
     ```

3. **Activate the Virtual Environment**
   - Activate the new virtual environment using:
     ```sh
     source 28/bin/activate
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
     python3 -m ipykernel install --user --name=28 --display-name "Python (lab 28)"
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
     TAVILY_API_KEY=
     OPENAI_KEY= 
     ```
   - note for generating TAVILY_API_KEY go here https://app.tavily.com/sign-in
   - OPENAI_KEY is not the azure openai key it is open ai key
         

9. **Start Jupyter Notebook**
   - Start Jupyter Notebook with the command:
     ```sh
     jupyter notebook
     ```

10. **Select the Correct Kernel**
    - At the top right corner of the Jupyter Notebook, click on the kernel and choose "Python (lab 28)".

11. **Verify the Environment**
    - To verify that you are in the correct environment, create a new cell and run the following command. If the output shows your current environment, you are all set:
      ```python
      import sys
      print(sys.executable)
      ```

12. **Run the Labs**
    - You are now ready to run any lab in Jupyter Notebook.