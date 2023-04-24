# HuggingFace-OpenAI install

### Step 1) Install Anaconda with the GUI installer

* Download the Anaconda installer for your operating system from [here](https://www.anaconda.com/products/individual#Downloads). We recommend using the graphical installer.
* In Ubuntu, you can simply do this
```shell
wget https://elephantscale-public.s3.amazonaws.com/downloads/Anaconda3-2023.03-Linux-x86_64.sh
chmod +x Anaconda3-2023.03-Linux-x86_64.sh
./Anaconda3-2023.03-Linux-x86_64.sh
```
* Last install step: say **"yes"** to initializing Conda

* If it still cannot recognize python, you can try this:

```shell
sudo apt install python-is-python3
```

### Step 2) Install Hugging Face

```bash
pip install transformers[torch]
```

### Step 3) Verify the installation

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```

* You should see something like this

```text
[{'label': 'POSITIVE', 'score': 0.9998656511306763}]
``` 

### Step 4) Wasn't that easy?

```
Yes!
```

#  Install Hugging Face datasets

### Step 5) Install datasets

```bash
pip install datasets
```

* Run the following command to check if 🤗 Datasets has been properly installed:

```bash
python -c "from datasets import load_dataset; print(load_dataset('squad', split='train')[0])"
```

* You should see something like this

![](../images/01.png)

