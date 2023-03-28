# Fine-tuning GPT3, GPT4

* In this lab we will practice fine-tuning GPT

### Lab Goals:

* **Note**
* OpenAI doesn't allow users to fine-tune GPT-3 directly; instead, they can use the OpenAI API to access GPT-3 and customize its behavior using API parameters.


### STEP 1) Login to the server

Each student is provided their individual server and credentials

(Instructor: use our ubuntu AMI, t2.large or t2.xlarge instances and Elasticsearch security group)

### STEP 2) Verify the environment

```bash
python --version
Python 3.7.3
```

### STEP 3) Get API access: 
* Request access to the OpenAI API at https://beta.openai.com/signup/. Once you have access, you'll receive an API key to authenticate your requests.

### STEP 4) Install the OpenAI Python library: 
* Install the official OpenAI Python library to interact with the API more easily. Use pip to install the package:

```shell
pip install openai
```

### STEP 5) Set up API authentication: 

* In your Python script or notebook, import the openai package and configure it with your API key:

```Python
import openai
openai.api_key = "your_api_key_here"
```

### STEP 6) Customize GPT-3 behavior: 
* 
  * While you can't fine-tune GPT-3 directly, you can customize its behavior using API parameters like `temperature`, `max_tokens`, `top_p`, and `prompt`. 
  * By adjusting these parameters, you can influence the generated text's creativity, length, and content.
  
* For example, to generate a text response from GPT-3 using the OpenAI API:

```python
response = openai.Completion.create(
    engine="davinci-codex",
    prompt="Write an introduction to the topic of climate change.",
    temperature=0.7,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)

generated_text = response.choices[0].text.strip()
print(generated_text)

```


* In this example, we're using the `davinci-codex` engine, a prompt related to climate change, and setting various parameters to influence the output. You can experiment with different parameter values to achieve the desired results.

* Please note that the availability of GPT-3, GPT-3.5, and GPT-4 and the process to fine-tune it keeps chaning. Consult the latest OpenAI documentation for any updates on GPT-3.5 and fine-tuning options: https://platform.openai.com/docs/.

* STEP 7) Bonus 

* For more details on prompt engineering with GPT-3, you can refer to the OpenAI Cookbook's guide on techniques to improve reliability: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_make_API_calls_more_reliable.ipynb

