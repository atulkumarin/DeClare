import numpy as np
import pandas as pd
import os
import csv

from torch.utils.data import Dataset

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
        
        print("Successfully read news data from {}".format(news_dataset_path))
        print("Number of articles = {}".format(self.news_df.shape[0]))
        print("Number of claims = {}".format(self.news_df.Claim_Source.nunique()))
        
        self.glove_df = pd.read_csv(glove_path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        
        self.vocab_path = os.path.join(os.path.dirname(news_dataset_path), 'vocab.npy')
        self.vocab_vectors_path = os.path.join(os.path.dirname(news_dataset_path), 'vocab_vectors.npy')
        
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """
        Builds the vocabulary for the loaded news dataset (all tokens/words in every claim and every article) 
        as well as the initial glove embeddings for this vocabulary
        """

        print("Building vocabulary for dataset")
        # If vocabulary was previously built, load it
        if os.path.isfile(self.vocab_path) and os.path.isfile(self.vocab_vectors_path):
            print("Found pre-built vocabulary.")
            self.vocab = np.load(self.vocab_path).item()
            self.initial_embeddings = np.load(self.vocab_vectors_path)
        
        else:
            print("Building vocabulary. This could take a while..")
            self.vocab = {}
            embeddings = []
            token_count = 0
            unk_encountered = False

            # Iterate over every token in the datasetsnopes = DeClareDataset(SNOPES_LOC, glove_data_file)
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
    
    def _vec(self, w):
        return self.glove_df.loc[w].as_matrix()
    
    def __len__(self):
        return len(self.news_df)
    
    def __getitem__(self, idx):
        data_item = self.news_df.iloc[idx]
        claim = data_item['Claim']
        article = data_item['Article']
        # TODO: convert claim and article to 1-hot vectors according to the vocabulary
        # TODO: decide representation for claim source and article source
        return