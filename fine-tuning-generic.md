# Fine-tuning an OpenAI model

* In this lab we will practice fine-tuning an OpenAI model

### Lab Goals:

* Install Elastic and verify its operation
* Prepare to use this installation instruction for all subsequent labs

### STEP 1) Login to the server

Each student is provided their individual server and credentials

(Instructor: use our ubuntu AMI, t2.large or t2.xlarge instances and Elasticsearch security group)

### STEP 2) Verify the environment

```bash
python --version
Python 3.7.3
```

### STEP 3) 

* Install OpenAI CLI

```shell
pip install --upgrade openai
```

### STEP 4) OPENAI_API_KEY

* Create your key on [OpenAI](https://openai.com/)

* Set your OPENAI_API_KEY environment variable by adding the following line into your shell initialization script (e.g. .bashrc, zshrc, etc.) or running it in the command line before the fine-tuning command:

###  STEP 5) Prepare training data

* Prepare data in the format below

```text
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
```

