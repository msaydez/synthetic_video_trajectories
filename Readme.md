# Synthetic Skier Trajectory Simulation

This repository contains the code used to generate **synthetic skier trajectories** using a cellular automaton model and to evaluate them against real data.

The project was developed for generating synthetic trajectory data for ski slope environments and evaluating its similarity to real-world trajectories.

---

# Files

### Simulation

This script runs the **cellular_automaton_multiple_aggressiveness.py** and generates synthetic skier trajectories.

The simulation models skier motion on a slope using:

- slope dynamics
- skier ability levels
- behavioral parameters
- multi-skier interactions

Running this script produces synthetic trajectories that can later be evaluated. The simulation is non-determinsitic.

---

### Evaluation

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

The generated trajectories used in the paper can be found [here] (https://1drv.ms/f/c/a3c853a721c97f9a/IgBju6Z2hhu6SY6Kxz0kH__-AZdBRZywGblksuwVTeH5KvA?e=6fGcFS)

- Use the create_pickle.py file to create pickle files for the training and validation sets.
- Run:
```  
python main.py --option 4 --min-len 60 --max-len 60 --batch-size 128 --lr 1e-4 --epochs 700 --target-len 60 --hidden-size 90 --model mamba --train --synthetic-only
```
Then
```  
python finetune.py --option 4 --min-len 60 --max-len 60 --batch-size 128 --lr 3e-4 --epochs 700 --target-len 60 --hidden-size 90 --model mamba --train --model_name NAME_OF_MODEL
```
Then
```
python tools/deep_eiou_yolov11.py --glide_weights NAME_OF_MODEL --glide_label NAME_OF_SAVE_FILE --in_dim 36 --num_freqs 8 --split test
```

Then
```
python TrackEval/scripts/run_mot_challenge.py --GT_FOLDER slope_track --BENCHMARK slope_track --METRICS HOTA CLEAR Identity --TRACKERS_FOLDER yolo11/slopetrack --USE_PARALLEL False --NUM_PARALLEL_CORES 1 --TRACKERS_TO_EVAL NAME_OF_SAVE_FILE
```


---


# Notes

- The repository focuses on **trajectory generation and evaluation** rather than full visual rendering.

- The generated trajectories are intended for **trajectory analysis and tracking research**.
