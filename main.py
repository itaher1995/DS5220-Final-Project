import ImageCaptionGenerator
import config
import numpy as np


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
    bestLoss = ImageCaptionGenerator.train(config.FILTER_SIZE_1,
                 config.NUM_FILTERS_1,
                 config.FILTER_SIZE_2, 
                 config.NUM_FILTERS_2, 
                 config.FILTER_SIZE_34, 
                 config.NUM_FILTERS_34, 
                 config.FILTER_SIZE_5, 
                 config.NUM_FILTERS_5,
                 config.STRIDE,
                 config.POOL_SIZE,
                 config.LEARNING_RATE)
    bestFS1 = config.FILTER_SIZE_1
    bestNF1 = config.NUM_FILTERS_1
    bestFS2 = config.FILTER_SIZE_2
    bestNF2 = config.NUM_FILTERS_2
    bestFS34 = config.FILTER_SIZE_34
    bestNF34 = config.NUM_FILTERS_34
    bestFS5 = config.FILTER_SIZE_5
    bestNF5 = config.NUM_FILTERS_5
    bestS = config.STRIDE
    bestEta = config.LEARNING_RATE
    
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
        loss = ImageCaptionGenerator.train(fs1,nf1,fs2,nf2,fs34,nf34,fs5,nf5,s,config.POOL_SIZE,e)
        if loss<bestLoss:
            bestFS1 = fs1
            bestNF1 = nf1
            bestFS2 = fs2
            bestNF2 = nf2
            bestFS34 = fs34
            bestNF34 = nf34
            bestFS5 = fs5
            bestNF5 = nf5
            bestS = s
            bestEta = e
    return bestFS1, bestNF1, bestFS2, bestNF2, bestFS34, bestNF34, bestFS5, bestNF5, bestS, bestEta

def main():
	#"""
	bestFS1, bestNF1, bestFS2, bestNF2, bestFS34, bestNF34, bestFS5, bestNF5, bestS, bestEta = randomSearch([3,5,9],[48,72],[3,5,9],[128,192],[2,3,5],[128,192],[3,5,9],[128,192],[2,3],[0.0001,0.0005,0.001,0.005,0.01])
	"""
	lr = ImageCaptionGenerator.test(config.FILTER_SIZE_1,
								config.NUM_FILTERS_1,
								config.FILTER_SIZE_2,
								config.NUM_FILTERS_2,
								config.FILTER_SIZE_34,
								config.NUM_FILTERS_34,
								config.FILTER_SIZE_5,
								config.NUM_FILTERS_5,
								config.STRIDE,
								config.POOL_SIZE,
								config.LEARNING_RATE)
	#"""
	return bestFS1, bestNF1, bestFS2, bestNF2, bestFS34, bestNF34, bestFS5, bestNF5, bestS, bestEta

if __name__ == "__main__":
	main()