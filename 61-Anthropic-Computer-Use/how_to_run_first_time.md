## Setup Local Environment

1. **Navigate to the Project Directory**
   - Go inside the lab folder.

2. **Create a Virtual Environment**
   - Run the following command to create a virtual environment:
     ```sh
     python3 -m venv myenv
     ```

3. **Activate the Virtual Environment**
   - Activate the new virtual environment using:
     ```sh
     source myenv/bin/activate
     ```
     
4. **Install Lab Requirements**
   - Install the lab requirements specified in the `requirements.txt` file:
     ```sh
     pip3 install -r requirements.txt

5. **Install IPython Kernel**
   - Install `ipykernel` to attach the Jupyter environment to the same kernel:
     ```sh
     pip3 install ipykernel
     ```

6. **Add Environment to Jupyter Kernel**
   - Add the current environment to the Jupyter kernel:
     ```sh
     python3 -m ipykernel install --user --name=lab --display-name "lab"
     ```

     ```

8. **Create .env File for API Keys**
    ```sh
     OPENAI_API_KEY=
     ```

9. **Start Jupyter Notebook**
   - Start Jupyter Notebook with the command:
     ```sh
     jupyter lab
     ```

10. **Select the Correct Kernel**
    - At the top right corner of the Jupyter Notebook, click on the kernel and choose "lab 50".

11. **Verify the Environment**
    - To verify that you are in the correct environment, create a new cell and run the following command. If the output shows your current environment, you are all set:
      ```python
      import sys
      print(sys.executable)
      ```

12. **Run the Labs**
    - You are now ready to run any lab in Jupyter lab.
