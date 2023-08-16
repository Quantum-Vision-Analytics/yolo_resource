import time
import torch
from utils.torch_utils import time_synchronized
from FileGenerator import FileGenerator
import threading
import abc
import argparse

class ModelInferenceHandler(abc.ABC):
    def __init__(self, options:argparse.ArgumentParser):
        self.fileGen = FileGenerator()
        if(options is not None):
            self.opt = options
        self.sleep_time = 0.25
        self.lock = threading.Lock()

    # Method that all the thread objects are running
    def ThreadWork(self, initWait:int):
        time.sleep(initWait)
        # Continue until all images are read into batches and the queue is empty
        # data_remained was ongoing
        while(self.batchQueue or self.data_remained):
            # Wait for the lock
            if self.batchQueue and not self.lock.locked():
                # Get a batch from queue and lock the queue object at that time
                self.lock.acquire()
                batch = self.batchQueue.pop(0)
                self.lock.release()

                # Process the batch
                with torch.no_grad():
                    batch = self.Preprocess(batch)
                    batch = self.Predict(batch)
                    self.Postprocess(batch) 
            else:
                time.sleep(self.sleep_time) # Add random interval

    @abc.abstractmethod
    def Train(self):
        pass
    
    # Loading weights-classes, assigning user arguments and setting up data loader
    @abc.abstractmethod
    def LoadResources(self,save_img=False):
        pass

    # Preprocessing images beforehand
    @abc.abstractmethod
    def Preprocess(self, batch:list):
        pass

    # Object detection and classification
    @abc.abstractmethod
    def Predict(self, batch:list):
        pass

    # Process results and save the labels
    @abc.abstractmethod
    def Postprocess(self, batch:list):
        pass

    # Start the whole process and manage the reading images, batches, batch_queue insertion parts
    @abc.abstractmethod
    def Detect(self):
        pass
