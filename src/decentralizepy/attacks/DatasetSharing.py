class DatasetSharing:
    def __init__(self, datasets):
        self.datasets = datasets
    
    def add_dataset(self, uid, dataset):
        self.datasets[uid] = dataset

    def get_dataset(self, uid):
        return self.datasets[uid]