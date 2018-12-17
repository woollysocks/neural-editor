# Neural Editor with Semi-Amortized Inference 

This codebase is forked from the source code for "[Generating Sentences by Editing Prototypes](https://arxiv.org/abs/1709.08878)" by Guu et al (2017). 

This project takes the Neural Editor and trains it using semi-amortized inference as described in [Semi-Amortized Variational Autoencoders](https://arxiv.org/abs/1802.02550) by Kim et al. (2018).


A detailed README for the original codebase can be found [here](https://github.com/kelvinguu/neural-editor/tree/readme). Note that this forked repo does not require using docker. 

The requirements needed for this code are listed in requirements.txt. Please note that this code is written in python v2.7 using pytorch v.0.1.12 (installation instructions are [here](https://pytorch.org/get-started/previous-versions/)).

Follow these steps to setup the codebase,
```
DATA_DIR=$HOME/neural-editor-data
REPO_DIR=$HOME/neural-editor

# Download repository
git clone https://github.com/kelvinguu/neural-editor.git $REPO_DIR

# Set up data directory
mkdir -p $DATA_DIR
cd $DATA_DIR

# Download word vectors
wget http://nlp.stanford.edu/data/glove.6B.zip  # GloVe vectors
unzip glove.6B.zip -d word_vectors

# Download expanded set of word vectors
cd word_vectors
wget https://worksheets.codalab.org/rest/bundles/0xa57f59ab786a4df2b86344378c17613b/contents/blob/ -O glove.6B.300d_yelp.txt
cd ..

# Download datasets into data directory
wget https://worksheets.codalab.org/rest/bundles/0x99d0557925b34dae851372841f206b8a/contents/blob/ -O yelp_dataset_large_split.tar.gz
mkdir yelp_dataset_large_split
tar xvf yelp_dataset_large_split.tar.gz -C yelp_dataset_large_split

# our code uses this variable to locate the data
export TEXTMORPH_DATA=$DATA_DIR
```

To train the model, run the following command,
```
$ python textmorph/edit_model/main.py configs/edit_model/edit_baseline.txt
```

or use `sb_train.sbatch`.
