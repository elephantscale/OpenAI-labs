If the virtual environment already exits, follow the below steps to run the notebook:

1. **Activate the Virtual Environment**
   - Activate the new virtual environment using:
     ```sh
     source 29/bin/activate
     ```
     
2. **Start Jupyter Notebook**
   - Start Jupyter Notebook with the command:
     ```sh
     jupyter notebook         
     ```
3. **Select the Correct Kernel**
    - At the top right corner of the Jupyter Notebook, click on the kernel and choose "Python (lab 29)".

4. **Verify the Environment**
    - To verify that you are in the correct environment, create a new cell and run the following command. If the output shows your current environment, you are all set:
      ```python
      import sys
      print(sys.executable)
      ```

5. **Run the Labs**
    - You are now ready to run any lab in Jupyter Notebook.