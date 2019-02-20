import numpy as np
import pandas as pd
import os
import csv

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class DeClareDataset(Dataset):
    """
    DeClare Fake News Detection Dataset
    """
    
    def __init__(self, news_dataset_path, glove_path):
        """        
        Arguments:
            news_dataset_path {string} -- path to the news dataset (.tsv file)
            glove_path {string} -- path to the glove pretrained .csv file
        """
        
        self.news_df = pd.read_csv(news_dataset_path, sep='\t', header=None)
        self.news_df.columns = ['Credibility', 'Claim_Source', 'Claim', 'Article', 'Article_Source']
        self.max_len_claim = self.news_df['Claim'].str.split().str.len().max()
        self.max_len_article = self.news_df['Article'].str.split().str.len().max()
        
        print("Successfully read news data from {}".format(news_dataset_path))
        print("Number of articles = {}".format(self.news_df.shape[0]))
        print("Number of claims = {}".format(self.news_df.Claim_Source.nunique()))
        
        self.glove_df = pd.read_csv(glove_path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        self.glove_dim = self.glove_df.shape[1]
        
        self.vocab_path = os.path.join(os.path.dirname(news_dataset_path), 'vocab.npy')
        self.vocab_vectors_path = os.path.join(os.path.dirname(news_dataset_path), 'vocab_vectors.npy')
        
        self._build_vocabulary()
        self._build_source_vocabularies()

        self.vocabulary_size = len(self.initial_embeddings)
        self.claim_source_vocab_size = len(self.claim_source_vocab)
        self.article_source_vocab_size = len(self.article_source_vocab)
    
    def _build_vocabulary(self):
        """
        Builds the vocabulary for the loaded news dataset (all tokens/words in every claim and every article) 
        as well as the initial glove embeddings for this vocabulary
        """
        
        # If vocabulary was previously built, load it
        if os.path.isfile(self.vocab_path) and os.path.isfile(self.vocab_vectors_path):
            print("Using pre-built vocabulary")
            self.vocab = np.load(self.vocab_path).item()
            self.initial_embeddings = np.load(self.vocab_vectors_path)
        
        else:
            print("Building vocabulary. This could take a while..")
            self.vocab = {}
            embeddings = []
            token_count = 0
            unk_encountered = False
            
            # Insert all zero embedding for <PAD> at position 0
            embeddings.append(np.zeros(self.glove_dim,))
            token_count += 1

            # Iterate over every token in the dataset
            for _, data_sample in self.news_df.iterrows():
                words_in_sample = data_sample['Claim'].split() + data_sample['Article'].split()
                for word in words_in_sample:
                    # Add new tokens to the vocabulary
                    if word not in self.vocab:
                        if word in self.glove_df.index:
                            # If the token exists in glove's database, load the glove weights
                            self.vocab[word] = token_count
                            token_count += 1
                            embeddings.append(self._vec(word))
                        else:
                            # Treat it as an unknown token
                            if not unk_encountered:
                                embeddings.append(self._vec('unk'))
                                unk_index = token_count
                                token_count += 1
                                unk_encountered = True
                            self.vocab[word] = unk_index
            
            
            self.initial_embeddings = np.array(embeddings)
            
            # Save the vocabulary
            np.save(self.vocab_path, self.vocab)
            np.save(self.vocab_vectors_path, self.initial_embeddings)
        
            print("Finished building vocabulary")
    
    def _build_source_vocabularies(self):
        """
        Builds the vocabulary for the claim sources and the article sources
        """

        self.claim_source_vocab = {}
        self.article_source_vocab = {}

        # Iterate over every news source in the dataset
        for _, data_sample in self.news_df.iterrows():
            claim_source = data_sample['Claim_Source']
            article_source = data_sample['Article_Source']
            if claim_source not in self.claim_source_vocab:
                self.claim_source_vocab[claim_source] = len(self.claim_source_vocab)
            if article_source not in self.article_source_vocab:
                self.article_source_vocab[article_source] = len(self.article_source_vocab)

    def _vec(self, w):
        return self.glove_df.loc[w].as_matrix()
    
    def __len__(self):
        return len(self.news_df)
    
    def __getitem__(self, idx):
        data_sample = self.news_df.iloc[idx]
        claim = data_sample['Claim']
        article = data_sample['Article']
        claim_source = data_sample['Claim_Source']
        article_source = data_sample['Article_Source']
        
        claim_word_indices = torch.tensor([self.vocab[key] for key in claim.split()], dtype=torch.long)
        claim_length = len(claim_word_indices)
        claim_word_indices = F.pad(claim_word_indices, (0, self.max_len_claim - claim_length), mode='constant', value=0)
        
        article_word_indices = torch.tensor([self.vocab[key] for key in article.split()], dtype=torch.long)
        article_length = len(article_word_indices)
        article_word_indices = F.pad(article_word_indices, (0, self.max_len_article - article_length), mode='constant', value=0)
        
        claim_source_index = torch.tensor(self.claim_source_vocab[claim_source], dtype=torch.long)
        article_source_index = torch.tensor(self.article_source_vocab[article_source], dtype=torch.long)

        return claim_word_indices, claim_length, article_word_indices, article_length, claim_source_index, article_source_index
