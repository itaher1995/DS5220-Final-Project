import ImageCaptionGenerator
import config


def idx_to_word_translate(idx_matrix, images):
    idx_to_word = pd.read_pickle("idx_to_word.pkl")

    #print(idx_matrix)

    new_caps = [[idx_to_word[idx] for idx in idx_cap] for idx_cap in idx_matrix]
    print(new_caps)



def main():
	ImageCaptionGenerator.train(config.FILTER_SIZE,config.NUM_FILTERS,config.STRIDES,config.POOL_SIZE,config.LEARNING_RATE)



if __name__ == "__main__":
	main()