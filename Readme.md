# Synthetic Skier Trajectory Simulation
---

## News

03/2026: Uploaded code for generating trajectories and trained models.

---

## Simulation

This script runs cellular automata model and generates synthetic skier trajectories. Running this script produces synthetic trajectories that can later be evaluated. The simulation is non-determinsitic.

```
python cellular_automaton_multiple_aggressiveness.py
```
---
## Data
- The generated trajectories used in the paper can be found [here](https://1drv.ms/f/c/a3c853a721c97f9a/IgBju6Z2hhu6SY6Kxz0kH__-AZdBRZywGblksuwVTeH5KvA?e=6fGcFS).
- Use create_pickle.py to create the pickle files needed.

## Training
- Go to the SlopeTrack website to download the dataset and their code for GlideTrack [website](https://slopetrack.github.io/).
- Include [train_syn.py](https://github.com/msaydez/synthetic_video_trajectories/blob/main/glide/train_syn.py) and [finetune.py](https://github.com/msaydez/synthetic_video_trajectories/blob/main/glide/finetune.py) in the **motion_training** folder.
- Run:
```  
python train_syn.py --option 4 --min-len 60 --max-len 60 --batch-size 128 --lr 1e-4 --epochs 700 --target-len 60 --hidden-size 90 --model mamba --train --synthetic-only
```
Then
```  
python finetune.py --option 4 --min-len 60 --max-len 60 --batch-size 128 --lr 1e-4 --epochs 300 --target-len 60 --hidden-size 90 --model mamba --train --model_name NAME_OF_MODEL
```

## Evaluation
- The same as the original SlopeTrack code.

## Trained Models
Download trained model [here](https://1drv.ms/u/c/a3c853a721c97f9a/IQDTzMQK7hbCS7dVGLkjL_jQAes0AbQ6qQdpo3kcpI4nu-A?e=hMgR4h).

## Acknowledgement  
The code is based on [SlopeTrack](https://github.com/SlopeTrack/Slope_Track). Thank you for your amazing work!

