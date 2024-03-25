# LLaMA-2 and the Czarnowska Templates

The code in this folder is used to produce predictions on the Czarnowska templates housed in `src/reference_implementations/fairness_measurement/resources/czarnowska_templates/`. These templates are described in detail in the readme in the directory above this folder and in the paper [here](https://aclanthology.org/2021.tacl-1.74/).

## Running the script

To run this script you'll need to request a GPU to run the script on. This is done by simply running the command

```bash
srun --gres=gpu:1 -c 4 --mem 16G -p t4v2 --pty bash
```

After being allocated a GPU, you'll need to __source the right environment__ before running the script. That is, before running the script `prompting_czarnowska_templates.py` make sure you run

```bash
source /ssd003/projects/aieng/public/prompt_engineering/bin/activate
```

The code is then run with
```bash
python -m src.reference_implementations.fairness_measurement.czarnowska_analysis.prompting_czarnowska_templates
```

The script uses the kaleidoscope tool to prompt the LLMs to perform sentiment inference on the Czarnowska templates file. Producing sentiment predictions for each sentence. We use 8-shot prompts drawn from the SST5 dataset.

__NOTE__: Making predictions on the entire Czarnowska templates takes a long time. We have already run OPT-6.7B, OPT-175B, LLaMA-2-7B, LLaMA-2-70B on these templates and stored the predictions in `src/reference_implementations/fairness_measurement/resources/predictions/` to visualize using the notebook `src/reference_implementations/fairness_measurement/fairness_eval_template.py`. If you would like to generate your own, please strongly consider pairing the templates down to only the groups you would like to investigate.

__NOTE__: The OPT predictions were done using a previously hosted model that is no longer available. There, we used 5-shot prompts and activation extraction to select the labels from generative distributions.
