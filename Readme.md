# Synthetic Skier Trajectory Simulation

This repository contains the code used to generate **synthetic skier trajectories** using a cellular automaton model and to evaluate them against real data.

The project was developed for generating synthetic trajectory data for ski slope environments and evaluating its similarity to real-world trajectories.

---

## Simulation

This script runs cellular automata model and generates synthetic skier trajectories. Running this script produces synthetic trajectories that can later be evaluated. The simulation is non-determinsitic.

```
python cellular_automaton_multiple_aggressiveness.py
```
---

## Evaluation

#### Trained Models
Download trained models [here](https://1drv.ms/f/s!App_ySGnU8ijvP5uIw1qva19CuLv_w?e=UPT23N). Put in folder named **pretrained**.

### Data
- The generated trajectories used in the paper can be found [here] (https://1drv.ms/f/c/a3c853a721c97f9a/IgBju6Z2hhu6SY6Kxz0kH__-AZdBRZywGblksuwVTeH5KvA?e=6fGcFS).
- The Slope-Track dataset can be found [here](https://1drv.ms/f/c/a3c853a721c97f9a/UgCaf8khp1PIIICjhj8PAAAAADl6ejU4H8z-u3A).
- Use create_pickle.py to create the pickle files needed.

### DeepEIoU + GlideTrack

1. Follow the installation instructions of [DeepEIoU](https://github.com/hsiangwei0903/Deep-EIoU).
   
2. Install SAHI and Ultralytics
~~~
pip install -U ultralytics sahi
~~~
3. Install Mamba
~~~
pip install mamba-ssm[causal-conv1d]
~~~



#### Training GlideTrack 
```
cd glide
```
- Run:
```  
python train_syn.py --option 4 --min-len 60 --max-len 60 --batch-size 128 --lr 1e-4 --epochs 700 --target-len 60 --hidden-size 90 --model mamba --train --synthetic-only
```
Then
```  
python finetune.py --option 4 --min-len 60 --max-len 60 --batch-size 128 --lr 1e-4 --epochs 300 --target-len 60 --hidden-size 90 --model mamba --train --model_name NAME_OF_MODEL
```

#### Testing on DeepEIoU
- Download [DeepEIoU](https://github.com/hsiangwei0903/Deep-EIoU)
- Replace their tracker folder with the one in the repository
- Put deep_eiou_yolov11.py in their tools folder

Run:
```
python tools/deep_eiou_yolov11.py --glide_weights NAME_OF_MODEL --glide_label NAME_OF_SAVE_FILE --in_dim 36 --num_freqs 8 --split test
```
Then use [TrackEval](https://github.com/JonathonLuiten/TrackEval)
```
python TrackEval/scripts/run_mot_challenge.py --GT_FOLDER slope_track --BENCHMARK slope_track --METRICS HOTA CLEAR Identity --TRACKERS_FOLDER yolo11/slopetrack --USE_PARALLEL False --NUM_PARALLEL_CORES 1 --TRACKERS_TO_EVAL NAME_OF_SAVE_FILE
```
---







