<h2 align="center">
  <b>SPARTUN3D: Situated Spatial Understanding of 3D World in Large Language Models</b>
</h2>

<!-- <h3 align="center">
ICLR 2025
</h3> -->

![overview](spartun3D.gif)

## Get Started
1. Clone Github repo.
```shell
git clone https://github.com/zhangyuejoslin/Spartun3D.git
cd Spartun3D
```
2. Following [LEO](https://github.com/embodied-generalist/embodied-generalist/blob/main/README.md?plain=1) to create a conda environment and install third-party libraries for point cloud backbones.

## Generating Data for Spartun3D

1. **Generate Metadata**  
   Run `scene_process.py` to collect raw spatial information.

2. **Generate Data with GPT-4o**  
   Run the code in `gpt_code` to generate data using GPT-4o. Please provide your OpenAI API key.

3. **Post-Process the Generated Data**  
   Use `data_process/3Rscan/gpt_code/post_process.py` to refine the generated data.

4. **Generated Captions and QA Pairs**  
   We provide the generated captions and question-answer pairs in `data_process/spartun3D_data`.  
   Please download the corresponding scene data from [LEO](https://huggingface.co/datasets/huangjy-pku/LEO_data/blob/main/3RScan-ours-align.zip).  
