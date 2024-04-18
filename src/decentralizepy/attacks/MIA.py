import numpy as np
import copy
import logging
import torch
from torch import nn

NUM_CLASSES = 10 # CIFAR10

class MIA():
    def __init__(self, model, dataset, datasets=None):
        self.model = model
        self.dataset = dataset
        self.datasets = datasets

    def compute_modified_entropy(self, p, y, epsilon=0.00001):
        """ Computes label informed entropy from 'Systematic evaluation of privacy risks of machine learning models' USENIX21 """
        assert len(y) == len(p)
        n = len(p)

        entropy = np.zeros(n)

        for i in range(n):
            pi = p[i]
            yi = y[i]
            for j, pij in enumerate(pi):
                if j == yi:
                    # right class
                    entropy[i] -= (1-pij)*np.log(pij+epsilon)
                else:
                    entropy[i] -= (pij)*np.log(1-pij+epsilon)

        return entropy


    def ths_searching_space(self, nt, train, test):
        """ it defines the threshold searching space as nt points between the max and min value for the given metrics """
        thrs = np.linspace(
            min(train.min(), test.min()),
            max(train.max(), test.max()), 
            nt
        )
        return thrs

    def mia_best_th(self, train_set, nt=150):
        """ Perfom naive, metric-based MIA with 'optimal' threshold """
        
        def search_th(Etrain, Etest):
            R = np.empty(len(thrs))
            for i, th in enumerate(thrs):
                tp = (Etrain < th).sum()
                tn = (Etest >= th).sum()
                acc = (tp + tn) / (Etrain.shape[0] + Etest.shape[0])
                R[i] = acc
            return R.max()
        
        # evaluating model on train and test set
        # I need loss, accuracy and Output
        _, Ltrain, Ptrain, Ytrain = self.testMIA(self.model, train_set)
        _, Ltest, Ptest, Ytest = self.testMIA(self.model, self.dataset.get_testset())
        
        # it takes a subset of results on test set with size equal to the one of the training test 
        n = Ptrain.shape[0]
        Ptest = Ptest[:n]
        Ytest = Ytest[:n]
        Ltest = Ltest[:n]
            
        # performs optimal threshold for loss-based MIA 
        thrs = self.ths_searching_space(nt, Ltrain, Ltest)
        loss_mia = search_th(Ltrain, Ltest)
        
        # computes entropy
        Etrain = self.compute_modified_entropy(Ptrain, Ytrain)
        Etest = self.compute_modified_entropy(Ptest, Ytest)
        
        # performs optimal threshold for entropy-based MIA 
        thrs = self.ths_searching_space(nt, Etrain, Etest)
        ent_mia = search_th(Etrain, Etest)
        
        return loss_mia, ent_mia


    # def mia_for_each_nn(self, modify_model, update_buffer):
    #     """ Run MIA for each attacker's neighbors """
        
    #     nn = sorted(list(update_buffer.keys()))
    #     model_copy = copy.deepcopy(self.model)

    #     # mias = np.zeros((len(nn), 2))
    #     mias = {}
    #     for i, v in enumerate(nn):
    #         modify_model(update_buffer, v, model_copy)
                        
    #         train_set = self.datasets.get_dataset(v).get_trainset()
            
    #         mias[i] = self.mia_best_th(train_set)
            
    #     return mias
    
    def mia_local(self):
        train_set = self.dataset.get_trainset()
        mias = self.mia_best_th(train_set)
        return mias
    
    def testMIA(self, model, testloader):
        """
        Function to evaluate model on the test dataset.

        Parameters
        ----------
        model : decentralizepy.models.Model
            Model to evaluate
        loss : torch.nn.loss
            Loss function to use

        Returns
        -------
        tuple(float, float)

        """
        model.eval()

        lossComplete = nn.CrossEntropyLoss(reduce=None, reduction='none')

        logging.debug("Test Loader instantiated.")

        correct_pred = [0 for _ in range(NUM_CLASSES)]
        total_pred = [0 for _ in range(NUM_CLASSES)]

        total_correct = 0
        total_predicted = 0

        y=[]
        p=[]
        l=[]

        with torch.no_grad():
            loss_val = 0.0
            count = 0
            for elems, labels in testloader:
                outputs = model(elems)
                loss_tmp = lossComplete(outputs, labels)
                # loss_val += loss_tmp.item()

                count += 1
                _, predictions = torch.max(outputs, 1)

                for label, prediction, output, loss_tmp in zip(labels, predictions, outputs, loss_tmp):
                    m = torch.nn.functional.softmax(output,dim=0)
                    
                    y.append(label.item())
                    p.append(m.numpy())
                    l.append(loss_tmp.item())
                    

                    if label == prediction:
                        correct_pred[label] += 1
                        total_correct += 1
                    total_pred[label] += 1
                    total_predicted += 1

        logging.debug("Predicted on the test set")

        for key, value in enumerate(correct_pred):
            if total_pred[key] != 0:
                accuracy = 100 * float(value) / total_pred[key]
            else:
                accuracy = 100.0
            logging.debug("Accuracy for class {} is: {:.1f} %".format(key, accuracy))

        accuracy = 100 * float(total_correct) / total_predicted
        loss_val = loss_val / count
        logging.info("Overall test accuracy is: {:.1f} %".format(accuracy))

        # raise ValueError(type(y), y[0], len(y), type(p), p[0], len(p), type(l), l[0], len(l))

        p=np.array(p)
        l=np.array(l)
        y=np.array(y)
        return accuracy, l, p, y