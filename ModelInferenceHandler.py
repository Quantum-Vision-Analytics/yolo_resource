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

    def ThreadWork(self, initWait:int):
        time.sleep(initWait)
        # Continue until all images are read into batches and the queue is empty
        while(self.batchQueue or self.ongoing):
            # Wait for the lock
            if self.batchQueue and not self.lock.locked():
                # Get a batch from queue and lock the queue object at that time
                self.lock.acquire()
                batch = self.batchQueue.pop(0)
                print("batch start")
                self.lock.release()

                # Process the batch
                with torch.no_grad():
                    #batch = self.Preprocess(batch)
                    batch = self.Predict(batch)
                    self.Postprocess(batch)
                print("batch finish")
            else:
                time.sleep(self.sleep_time) # Add random interval
        print("while is finsihed")

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

    def Detect(self):
        # Prepare for detection
        t0 = time_synchronized()
        self.LoadResources()
        t1 = time_synchronized()
        batch_size = self.opt.batch_size if self.opt.batch_size > 1 else 5
        self.threadCount = self.opt.thread_count if self.opt.thread_count > 0 else 2
        lastIndex = self.dataset.nf
        self.ongoing = True
        threadList = []
        self.batchQueue = []
        batch_buffer_limit = 3 # int(self.threadCount / 3) # Adjust
        
        for i in range(self.threadCount):
            threadList.append(threading.Thread(target=self.ThreadWork, args=(i * self.sleep_time + 1,)))
            threadList[i].start()

        # Iterate per image
        nextIndex = 0
        while(self.ongoing):
            # Wait if hit the batch buffer limit
            if len(self.batchQueue) == batch_buffer_limit:
                time.sleep(self.sleep_time)
                continue

            # Preparing batches
            nextIndex += batch_size
            batch = []

            if(nextIndex < lastIndex):
                for x in range(batch_size):
                    batch.append(self.dataset.__next__())
                self.lock.acquire()
                self.batchQueue.append(batch)
                self.lock.release()

            else:
                for x in range(batch_size - (nextIndex - lastIndex)):
                    batch.append(self.dataset.__next__())
                    self.ongoing = False
                self.lock.acquire()
                self.batchQueue.append(batch)
                self.lock.release()

        # Wait for all threads to finish
        for x in reversed(range(len(threadList))):
            if threadList[x].is_alive():
                threadList[x].join()
        
        del threadList

        # Output timers
        t2 = time_synchronized()
        print(f"Loading model: {round(t1-t0,3)} seconds\nInference: {round(t2-t1,3)} seconds\nTotal time: {round(t2-t0,3)} seconds")

        # Create classes file
        if self.save_txt:
            self.fileGen.Generate_Classes(str(self.save_dir), self.names)