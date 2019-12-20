# Verified Summarization

If this code is helpful in your research, please cite the following publication

> Ashish Sharma, Koustav Rudra, and Niloy Ganguly. "Going Beyond Content Richness: Verified Information Aware Summarization of Crisis-Related Microblogs." Proceedings of the 28th ACM International Conference on Information and Knowledge Management. ACM CIKM, 2019.


## Initial Steps

### Dataset

- We use the dataset created by Zubiaga et al, 2017 which can be downloaded from [here](https://figshare.com/articles/PHEME_dataset_of_rumours_and_non-rumours/4010619). 


### POS TAGGING

- Use [GATE Twitter Part-of-Speech Tagger](https://gate.ac.uk/wiki/twitter-postagger.html)

- Command: 

```
$ java -jar twitie_tag.jar models/gate-EN-twitter.model $input_file > $output_file
```

- **$input_file**: File with each line containing a tweet (only text - space separated words) 
- **$output_file**: space separated <word>_<TAG> for each tweet


### Processing Corpus File 

Command:
```
$ python create-corpus-file.py
```

Configure the following input variables inside the code:

- **src_data_path**: Folder containing files of Source Tweets. In every file, a line contains - DateTime, TweetId, UserId, Tweet Text, Rumor Tag (tab-separated)
- **rep_data_path**: Folder containing files of Reply Tweets. In every file, a line contains - DateTime, TweetId, UserId, Tweet Text, Source Tweet Id (tab-separated)
- **src_pos_tag_path**: Folder containing Source POS Tags files. In every file, a line contains - TweetId, Output of Twitie (space separated <word>_<TAG>) (space-separated)
- **rep_pos_tag_path**: Folder containing Reply POS Tags files. In every file, a line contains - TweetId, Output of Twitie (space separated <word>_<TAG>) (space-separated)
- **tentative_path**: File containing LIWC list of tentative words
- **certain_path**: File containing LIWC list of certainty words
- **negate_path**: File containing LIWC list of negative words
- **question_path**: File containing list of question words


Configure the following output variables inside the code:

- **corpus_path**: Corpus file (will be input to the topic model). Each line in the file contains: TweetId, Content-Words (space-separated), Expression-Words(Space-Separated), TweetType(S/R), Time(0-1)


## Content-Expression Topic Model (CETM)

Compile:

```
$ g++ -std=c++11 topic-model.cpp -o model
```

Run:

```
$ ./model K T iter
```
where:

```
K:    Number of Content Word Topics
T:    Number of Expression Word Topics
iter: Number of iterations to run the model for
```

Configure the following input variables inside the code:

- **corpus_path**: Corpus file created in the previous step (preprocessing corpus file).

Configure the following output variables inside the code:

- **destpath**: Folder where all the output files will be stored


Description of files inside the destination folder is as follows:

- **c-vocab-mapping.txt**: Content words to indices mapping.
- **e-vocab-mapping.txt**: Expression words to indices mapping.
- **behavior-mapping.txt**: Tweet Type to indices mapping. 
- **topic-priors.txt**: Prior probability of content topics.
- **expression-priors.txt**: Prior probability of expression topics.
- **c-topic-word-distribution.txt**: Content Topic to Word Distribution.
- **e-topic-word-distribution.txt**: Expression Topic to Word Distribution.
- **topic-behavior-distribution.txt**: Topic to Behavior Distribution.
- **table-assignment-status.txt**: Status of Data points seating.
- **top-c-topic-words.txt**: Top 20 words in each content-word topic.
- **top-e-topic-words.txt**: Top 20 words in each expression-word topic.
- **e-topic-time-alpha.txt**: Expression-Topic-Time Alpha values.
- **e-topic-time-beta.txt**: Expression-Topic-Time Beta values.
- **c-topic-time-alpha.txt**: Content-Topic-Time Alpha values.
- **c-topic-time-beta.txt**: Content-Topic-Time Beta values.


## Computing Tweet Posteriors

Command:

```
$ python compute-posteriors.py
```


Configure the following input variables inside the code:

- **basepath**: Folder created by topic-model.cpp
- **CORPUS_PATH**: Corpus file created.

Configure the following output variables inside the code:

- **POSTERIOR_PATH**: File where posteriors (probability vectors) for each tweet will be stored. 

## Verified Tweet Detection using Tree LSTM

### Generating Trees

Command:

```
$ python generate-trees.py
```

Configure the following input variables inside the code:

- **datapath**: The original dataset folder (download from [here](https://figshare.com/articles/PHEME_dataset_of_rumours_and_non-rumours/4010619)) 
- **feature_path**: File containing input feature vectors for all tweets in the dataset. The file contains two tab-separated columns - tweet_id, features
- **output_path**: Path of the folder where you want the generated trees to be stored

Each tree is stored as a dictionary. A sample tree and the corresponding stored dictionary is shown below:

![Tree-Example](Tree-Ex.png?raw=true "Tree_Example") 

```
tree = {
        'f': [0.234, .... , ], 'l': [0, 1], 'c': [
            {'f': [0.109, ... , ], 'l': [0, 1], 'c': []},
            {'f': [0.712, ... , ], 'l': [0, 1], 'c': [
                {'f': [0.352, ... , ], 'l': [0, 1], 'c': []}
            ]},
        ],
    }
```

Here, f is the input feature vector for each node of the tree, l is the true label of the root of the tree stored as a 2-dimensional one-hot vector (dim-1: verified, dim-2: unverified), and c is a list of children of a node. 

### Training and Testing Tree-LSTM

Command:

```
$ python train-Tree-LSTM.py
```

Configure the following input variables inside the code:

- **tree_path**: Path to the folder containing generate trees (output_path of the last step).
- **IN_FEATURES**: Size of the input feature vectors
- **NUM_ITERATIONS**: Number of iterations for training
- **BATCH_SIZE**: Batch size for training
- **test_set**: Disaster events on which you want to test.


## VERISUMM

Code coming up soon!!
