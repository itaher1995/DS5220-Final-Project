import ImageCaptionGenerator
import config


def idx_to_word_translate(idx_matrix, images):
    idx_to_word = pd.read_pickle("idx_to_word.pkl")

    #print(idx_matrix)

    new_caps = [[idx_to_word[idx] for idx in idx_cap] for idx_cap in idx_matrix]
    print(new_caps)



def main():
	#"""
	lr = ImageCaptionGenerator.train(config.FILTER_SIZE_1,
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


if __name__ == "__main__":
	main()