# Neural Editor with Semi-Amortized Inference 

This codebase is forked from the source code for "[Generating Sentences by Editing Prototypes](https://arxiv.org/abs/1709.08878)" by Guu et al (2017). 

This project takes the Neural Editor and trains it using semi-amortized inference as described in [Semi-Amortized Variational Autoencoders](https://arxiv.org/abs/1802.02550) by Kim et al. (2018)


A detailed README for the original codebase can be found [here](https://github.com/kelvinguu/neural-editor/tree/readme). We recommend using that README to help set up the codebase, though this forked repo does not require using docker. 

To train the model, run the following command,
```
$ python textmorph/edit_model/main.py configs/edit_model/edit_baseline.txt
```