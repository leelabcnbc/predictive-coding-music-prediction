This repository contains the implementation for the paper:

## [**Relating Human Perception of Musicality to Prediction in a Predictive Coding Model**](https://arxiv.org/abs/2210.16587)

Nikolas McNeal*, Jennifer Huang*, Aniekan Umoren, Shuqi Dai, Roger Dannenberg, Richard Randall, Tai Sing Lee
 
## Introduction
<p align="center">
<img src="/img/fig_1_submission.jpeg" width=60% height=60%>
</p>
We explore the use of a neural network inspired by predictive coding for modeling human music perception. This network was developed based on the computational neuroscience theory of recurrent interactions in the hierarchical visual cortex. When trained with video data using self-supervised learning, the model manifests behaviors consistent with human visual illusions. Here, we adapt this network to model the hierarchical auditory system and investigate whether it will make similar choices to humans regarding the musicality of a set of random pitch sequences. When the model is trained with a large corpus of instrumental classical music and popular melodies rendered as mel spectrograms, it exhibits greater prediction errors for random pitch sequences that are rated less musical by human subjects. We found that the prediction error depends on the amount of information regarding the subsequent note, the pitch interval, and the temporal context. Our findings suggest that predictability is correlated with human perception of musicality and that a predictive coding neural network trained on music can be used to characterize the features and motifs contributing to human perception of music.

## Install

### Setup
To build the music-prediction conda environment, run the following commands:

```
conda env create -f environment.yml
conda activate music-prediction
```



### Train
To train on the Medley-solos-DB dataset, first download the Medley-solos-DB dataset from [Zenodo](https://zenodo.org/record/1344103#.Y2Bwe-zML0o) and place it in `./train_data`.

If desired, change any parameters or the name of the saved model in the train file, and run ```python train.py```

Alternatively, use a pre-trained model from the ```models/``` folder. The model used in the paper is ```prednet-4layer-3seconds.pt```, which is a 4-layer PredNet trained on 3-second clips from the Medley-solos-DB dataset.


### Test

To obtain the average prediction error for each test sequence, run ```python test.py```. The average MSE for each test sequence will be output in the console.

The test file will also save ```mses.npy```, which contains the prediction error of every frame of every sequence. 

```mses[0]``` returns the ordered names of the test sequences

```mses[1]``` returns the errors of all 50 * n sequences, where n is the number of shifted runs as determined by `rng` in the test file (default=1)

```mses[1][i]``` returns the errors of the ith sequence

```mses[1][i][j]``` returns the error of the ***j***th frame in the ***i***th sequence


A visualization of our approach to obtaining a resolution of one time-unit is available in the spreadsheet ```test-results.xlsx```. Data from `mses.npy` is input into the first nine ten sheets, and the last sheet demonstrates the computations to obtain the context effect (normalized non-musical error minus normalized musical error).


## Citation

Our work is currently under peer review. If you found our work useful, please consider citing our pre-print:
```
@misc{mcneal2022relating,
      title={Relating Human Perception of Musicality to Prediction in a Predictive Coding Model}, 
      author={McNeal, Nikolas and Huang, Jennifer and Umoren, Aniekan and Dai, Shuqi and Dannenberg, Roger and Randall, Richard and Lee, Tai-Sing},
      year={2022},
      eprint={2210.16587},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

## Contact

If you have any questions, please contact the corresponding author, Tai Sing Lee, through email:
taislee@andrew.cmu.edu




