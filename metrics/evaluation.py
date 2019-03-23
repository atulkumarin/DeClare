import torch
import numpy as np

class DeClareEvaluation(object):

    def __init__(self, model, test_dataloader, device='cpu'):
        self.model = model
        self.dataloader = test_dataloader
        self.device = device

        self.test_data, self.predictions = self._predict_test()

    def _predict_test(self):
        with torch.no_grad():
            dataiter = iter(self.dataloader)
            data_sample = dataiter.next()

            idx = np.argsort(-data_sample[3])
    
            for i in range(len(data_sample)):
                data_sample[i] = data_sample[i][idx].to(self.device)
            
            out = self.model(data_sample[0], data_sample[1], data_sample[2], data_sample[3], data_sample[4], data_sample[5])
        
        return data_sample, out
        
    def claim_wise_accuracies(self):
        claims = self.test_data[0]
        unique_claims = torch.unique(claims, dim=0)
        per_claim_score = torch.zeros_like(unique_claims).float()

        for i, claim in enumerate(unique_claims):
            indices = (claims == claim).all(dim=1).nonzero().squeeze(-1)
            claim_scores = self.predictions[indices]
            per_claim_score[i] = claim_scores.mean()
        
        return per_claim_score
    