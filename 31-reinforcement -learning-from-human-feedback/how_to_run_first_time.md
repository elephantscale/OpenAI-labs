## Setup Local Environment for lab 31: reinforcement -learning-from-human-feedback

1. **Navigate to the Project Directory**
   - Go inside the `31-reinforcement -learning-from-human-feedback` folder.

2. **Create a Virtual Environment**
   - Run the following command to create a virtual environment:
     ```sh
     python3.10 -m venv 31
     ```

3. **Activate the Virtual Environment**
   - Activate the new virtual environment using:
     ```sh
     source 31/bin/activate
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
     python3.10 -m ipykernel install --user --name=31 --display-name "Python (lab 31)"
     ```

7. **Install Lab Requirements**
   - Install the lab requirements specified in the `requirements.txt` file:
     ```sh
     pip3 install -r requirements.txt
     ```

8. **Start Jupyter Notebook**
   - Start Jupyter Notebook with the command:
     ```sh
     jupyter lab 
     ```
9. **Select the Correct Kernel**
    - At the top right corner of the Jupyter Notebook, click on the kernel and choose "Python (lab 30)".

10. **Verify the Environment**
    - To verify that you are in the correct environment, create a new cell and run the following command. If the output shows your current environment, you are all set:
      ```python
      import sys
      print(sys.executable)
      ```
11. **Run the Labs**
    - You are now ready to run any lab in Jupyter Notebook.