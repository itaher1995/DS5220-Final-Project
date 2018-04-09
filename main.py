import ImageCaptionGenerator
import config
import numpy as np
import pandas as pd
import os


def idx_to_word_translate(idx_matrix, images):
    idx_to_word = pd.read_pickle("idx_to_word.pkl")

    #print(idx_matrix)

    new_caps = [[idx_to_word[idx] for idx in idx_cap] for idx_cap in idx_matrix]
    print(new_caps)

def randomSearch(filterSize_1,
        numFilters_1,
        filterSize_2,
        numFilters_2,
        filterSize_34,
        numFilters_34,
        filterSize_5,
        numFilters_5,
        strides,
        eta):
    LossCSV = []
    # bestResults = ImageCaptionGenerator.train(config.FILTER_SIZE_1,
    #              config.NUM_FILTERS_1,
    #              config.FILTER_SIZE_2, 
    #              config.NUM_FILTERS_2, 
    #              config.FILTER_SIZE_34, 
    #              config.NUM_FILTERS_34, 
    #              config.FILTER_SIZE_5, 
    #              config.NUM_FILTERS_5,
    #              config.STRIDE,
    #              config.POOL_SIZE,
    #              config.LEARNING_RATE)
    # fs1 = config.FILTER_SIZE_1
    # nf1 = config.NUM_FILTERS_1
    # fs2 = config.FILTER_SIZE_2
    # nf2 = config.NUM_FILTERS_2
    # fs34 = config.FILTER_SIZE_34
    # nf34 = config.NUM_FILTERS_34
    # fs5 = config.FILTER_SIZE_5
    # nf5 = config.NUM_FILTERS_5
    # s = config.STRIDE
    # e = config.LEARNING_RATE
    # LossCSV.append(bestResults)

    for i in range(5):
        fs1 = np.random.choice(filterSize_1)
        fs2 = np.random.choice(filterSize_2)
        fs34 = np.random.choice(filterSize_34)
        fs5 = np.random.choice(filterSize_5)
        nf1 = np.random.choice(numFilters_1)
        nf2 = np.random.choice(numFilters_2)
        nf34 = np.random.choice(numFilters_34)
        nf5 = np.random.choice(numFilters_5)
        s = np.random.choice(strides)
        e = np.random.choice(eta)
        results = ImageCaptionGenerator.train(fs1,nf1,fs2,nf2,fs34,nf34,fs5,nf5,s,config.POOL_SIZE,e)
        LossCSV.append(results)
    df=pd.DataFrame.from_records(LossCSV)
    df.to_csv('randomSearchTrain_TAHER.csv',index=False)
    

def main():
	#"""
	randomSearch([3,5,9],[48,72],[3,5,9],[128,192],[2,3,5],[128,192],[3,5,9],[128,192],[2,3],[0.0001,0.0005,0.001,0.005,0.01])
	"""
    filenames = os.listdir('pretrained_models') #make this final layer
    for file in filenames:
        params = file.split("-")
        print(params)
    #filterSize_1,numFilters_1,filterSize_2,numFilters_2,filterSize_34,numFilters_34,filterSize_5,numFilters_5,strides,eta = params
	# lr = ImageCaptionGenerator.test(filterSize_1,
 #        numFilters_1,
 #        filterSize_2,
 #        numFilters_2,
 #        filterSize_34,
 #        numFilters_34,
 #        filterSize_5,
 #        numFilters_5,
 #        strides,
 #        eta)
	#"""
	
def mainTest():
    summaryResults=[]
    runningBLEU=0
    for folder in os.listdir('pretrained_models_TAHER'):
        hyperparameters = folder.split('-')
        hyperparameters = [int(x) if float(x)>=1 else float(x) for x in hyperparameters]
        print(hyperparameters)
        for i in range(1):
            summary=ImageCaptionGenerator.test(hyperparameters[0],
        hyperparameters[1],
        hyperparameters[2],
        hyperparameters[3],
        hyperparameters[4],
        hyperparameters[5],
        hyperparameters[6],
        hyperparameters[7],
        hyperparameters[8],
        hyperparameters[9],
        hyperparameters[10])
            if i==0:
                summaryResults.append(summary)
            runningBLEU += summary['BLEU_Score']
        summaryResults[-1]['BLEU_Score']=runningBLEU/1
    df=pd.DataFrame.from_records(summaryResults)
    df.to_csv('summaryResults-Test.csv',index=False)
        

if __name__ == "__main__":
    #main()
    mainTest()