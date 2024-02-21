# Prompt Engineering Laboratory

This repository holds all of the code associated with the project considering prompt engineering for large language models (LLMs). This includes work around reference implementations and demo notebooks.

The static code checker and all implementations run on python3.9

All reference implementations are housed in `src/reference_implementations/`. Datasets to be used are placed in the relevant resources subfolder for the implementation.

## A Few Tips for Getting Started

1. As part of your cluster account, you have been allocated a scratch folder where checkpoints, training artifacts, and other files may be stored. It should be located at the path:

     `/scratch/ssd004/scratch/<cluster_username>`

    If you don't see this path, please let your facilitator know, and we will ensure that it exists. For example, it may have been created on another `ssd` partition, rather than 004 etc.

2. We have a pre-constructed environment for running experiments and using the notebooks in this repository. This environment is housed in the public path `/ssd003/projects/aieng/public/prompt_engineering`

    __NOTE__ You do not have permission to modify the environment. It should contain all of the libraries necessary to run the notebooks and experiments in this repository.

    However, if you need to add something to this environment, you have two options.

    1) You may create your own virtual environment and install the dependencies using the requirements.txt file at the top level of this repository. You will be able to add any libraries you would like to this new venv.

    2) You may request that a library be added to the `prompt_engineering` environment. This will generally only be available as an option to fix bugs, as changing the environment will affect everyone.

3. We have provided some exploration guidance in the markdown `Exploration_Guide.md`. This guide provides some suggestions for exploration for each hands-on session based on the concepts covered in preceding lectures.

    __Note__: This guide is simply a suggestion. You should feel free to explore whatever is most interesting to you.

Below is a brief description of the contents of each folder in the reference implementations directory. In addition, each directory has at least a few readmes with more in-depth discussions. Finally, many of the notebooks are heavily documented.

This repository is organized as follows

## Prompting LLM On the Cluster through Kaleidoscope

These reference implementations are housed in `src/reference_implementations/prompting_vector_llms/`

This folder contains notebooks and implementations of prompting LLMs hosted on Vector's compute cluster. There are notebooks for demonstrating various prompted downstream tasks, the affects of prompts on tasks like aspect-based sentiment analysis, text classification, summarization, and translation, along with prompt ensembling, activation fine-tuning, and experimenting with whether discrete prompts are transferable across architectures. More details are found in `src/reference_implementations/prompting_vector_llms/README.MD`

## Fairness in language models

These reference implementations reside in `src/reference_implementations/fairness_measurement/`

This folder contains implementations for measuring fairness for LMs. There is an implementation that assesses fairness through fine-tuning or prompting to complete a sentiment classification task. We also consider LLM performance on the [CrowS-Pairs](https://aclanthology.org/2020.emnlp-main.154/) task and the [BBQ](https://aclanthology.org/2022.findings-acl.165/) task as a means of probing model bias and fairness.

## Hugging Face Basics

These reference implementations are in `src/reference_implementations/hugging_face_basics/`.

The reference implementations here are of two kinds. The first is a collection of examples of using HuggingFace for basic ML tasks. The second is a discussion of some important metrics associated with NLP, specifically generative NLP.

## Koala Language Model

In the notebook `src/reference_implementations/prompting_vector_llms/llm_prompting_examples/llm_prompt_ift_koala_local.ipynb`, you can load Koala 7B, an instruction fine-tuned model. While this model is generally less performant than Falcon or LLaMA-2, it has been instruction fine-tuned. Thus, it is sometimes better at following direct instructions than these new models and worth some exploration.

## Accessing Compute Resources

### Virtual Environments

As mentioned above, we have a pre-built environment for running different parts of the code.

* `/ssd003/projects/aieng/public/prompt_engineering`

If you're using a notebook launched through Jupyter hub, you can simply select the `prompt_engineering` kernel from the dropdown.

Before manually starting a notebook on the cluster or running code, you should source this environment with the command:
```bash
source /ssd003/projects/aieng/public/prompt_engineering/bin/activate
```

When using the pre-built environments, you are not allowed to pip install to them. If you would like to setup your own environments, see section `Installing Custom Dependencies`.

### Starting a Notebook through Jupyter Hub

If connected to the VPN, you should have access to the Vector Jupyter Hub, which makes launching notebooks easy. __NOTE__: Notebooks launched using this hub are backed by a T4V2 GPU. If, for whatever reason, you need something bigger, you'll need to follow the guide below for launching a notebook on a reserved GPU node.

0) Prior to connecting to a notebook, you have to have cloned the Prompt Engineering Lab repository on the cluster. To do so, log into the cluster through your terminal and follow the instructions in `Github_Basics.md` if you have not done so already as part of the cluster on-boarding.

1) Connect to the Vector VPN.

2) Navigate to the [Jupyter Hub](https://vdm1.vectorinstitute.ai:8000/hub/login?next=%2Fhub%2F). Your login credentials are the same as your cluster credentials.

3) Follow the prompts to start your notebook. You should be able to see your cloned Prompt Engineering Laboratory repository in your file tree. From there you should be able to open your notebook of interest.

4) __Make sure you select the `prompt_engineering` kernel from the drop down before running any code.__

5) When exiting a notebook, make sure you close and shutdown your session. Otherwise you are likely to get CUDA Out-of-Memory exceptions when you start another notebook.


### Launching an interactive session on a GPU node and connecting VSCode Server/Tunnel.

From any of the v-login nodes, run the following. This will reserve an A40 GPU and provide you a terminal to run commands on that node.

```bash
srun --gres=gpu:1 -c 8 --mem 16G -p a40 --pty bash
```

Note that `-p a40` requests an a40 gpu. You can also access smaller `t4v2` and `rtx6000` gpus this way. The `-c 8` requests 8 supporting CPUs and `--mem 16G` request 16 GB of cpu memory.

### Starting a Notebook from a GPU Node.

Once an interactive session has been started, we can run a jupyter notebook on the gpu node.

Below, we start the notebook on the example port `8888`: If the port `8888` is taken, try another random port between 1024 and 65000.
Also note the URL output by the command to be used later. (ex. http://127.0.0.1:8888/?token=7ba0ba5c3e9f5668f92518e4c5e723fea8b69aca065b4d57)

```bash
jupyter notebook --ip 0.0.0.0 --port 8888
```

Using a new terminal window from our personal laptop, we need to create an ssh tunnel to that specific port of the gpu node:
Note that `gpu001` is the name of the gpu we reserved at the beginning. Remember that the port needs to be the same as your jupyter notebook port above.
```bash
ssh username@v.vectorinstitute.ai -L 8888:gpu001:8888
```

Keep the new connection alive by starting a tmux session in the new local terminal:
```bash
tmux
```

Now we can access the notebooks using our local browser. Copy the URL given by the jupyter notebook server into your local web browser:
```bash
(Example Token)
http://127.0.0.1:8888/?token=7ba0ba5c3e9f5668f92518e4c5e723fea8b69aca065b4d57
```

You should now be able to navigate to the notebooks and run them.

**Don't close the local terminal windows in your personal laptop!**

### Connecting a VSCode Server and Tunnel on that GPU Node

Rather than working through hosted Jupyter Notebooks, you can also connect directly to a VS Code instance on the GPU. After the cluster has fulfilled your request for a GPU session, run the following to set up a VSCode Server on the GPU node.

This command downloads and saves VSCode in your home folder on the cluster. You need to do this only once:
```bash
cd ~/

curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz

tar -xf vscode_cli.tar.gz
rm vscode_cli.tar.gz
```

Please verify the beginning of the command prompt and make sure that you are running this command from a __GPU node__ (e.g., `user@gpu001`) and not the login node (`user@v`). After that, you can spin up a tunnel to the GPU node using the following command:

```bash
~/code tunnel
```

You will be prompted to authenticate via Github. On the first run, you might also need to review Microsoft's terms of services. After that, you can access the tunnel through your browser. If you've logged into Github on your VSCode desktop app, you can also connect from there by installing the extension `ms-vscode.remote-server`, pressing Shift-Command-P (Shift-Control-P), and entering `Remote-Tunnels: Connect to Tunnel`.

Note that you will need to keep the SSH connection running while using the tunnel. After you are done with the work, stop your session by pressing Control-C.

## Installing Custom Dependencies

__Note__: The following instructions are for anyone who would like to create their own python virtual environment to run your own experiments that require libraries not already installed in `prompt_engineering`. If you would just like to run the code you can use one of our pre-built virtual environments by following the instructions in Section `Virtual Environments`, above.

### Virtualenv installation

If you want to create your own environment then you can do so by creating your own virtual environment with the command
```bash
python -m venv <name_of_venv>
```
then
```bash
source <name_of_venv>/bin/activate
```
To install all of the packages in the `prompt_engineering` environment, you can simply run
```bash
pip install -r requirements.txt
```
from the top level of the repository. The environment is yours locally and you are free to modify it as you wish.

## A note on disk space

As discussed above, you have a local path with 50GB of disk space at
```
`/scratch/ssd004/scratch/<cluster_username>`
```
or the like. It is unlikely that you fill this space. However, if you decide to save a lot of information in your directory, it is possible. If the space fills, you may be unable to run notebooks or jobs. Please be cognizant of the space and clean up old data if you begin to fill the directory.

## Using Pre-commit Hooks (for developing in this repository)

To check your code at commit time
```
pre-commit install
```

You can also get pre-commit to fix your code
```
pre-commit run
```
