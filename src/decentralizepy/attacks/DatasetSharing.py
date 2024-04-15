class DatasetSharing:
    def __init__(self, datasets, neighboursets=None):
        self.datasets = datasets
        self.neighboursets = neighboursets
    
    def add_dataset(self, uid, dataset):
        self.datasets[uid] = dataset

    def get_dataset(self, uid):
        return self.datasets[uid]
    
    def add_neighbourset(self,uid,neighbourset):
        self.neighboursets[uid] = neighbourset
    
    def get_neighbourset(self,uid):
        return self.neighboursets[uid]