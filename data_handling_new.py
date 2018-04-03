
import pandas as pd
import numpy as np
from time import time
from skimage import data, transform
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import Counter
import config
import pickle
import os

def get_data(file):

	filename = "/Users/forresthooton/Documents/Masters Classes/Supervised Machine Learning/Class Project/" + file
	data = pd.read_csv(filename)

	return data

def clean_string(string):
	string = string.lower().strip('.')
    
	return string

def get_tokens(data):
	captions = []
	for idx, row in data.iterrows():
		value = row['caption']
		captions.append(clean_string(value))  # Why doesn't this strip all '.'s??
            #print(value.lower().strip('.;'))
    
	joined_cap = " ".join(captions) # Join all captions into one string
    
	tokenized_cap = joined_cap.split()  # Split words individually

	return tokenized_cap

def build_vocab(filetype):
	filename = filetype + "_annotations.csv"
	data = get_data(filename)

	words = get_tokens(data)

	word_counts = Counter(words)
	
	print(len(word_counts))

	top_words = word_counts.most_common(config.NUM_TOKENS)

	words_for_dict = []

	for word, count in top_words:
		words_for_dict.append(word)

	print(len(words_for_dict))

	return words_for_dict

def write_captions(filetype,dictionary):
	filename = filetype + "_annotations.csv"
	data = get_data(filename)

	for idx, row in data.iterrows():
		ann_words = word_tokenize(row['caption'])
		print(ann_words)

def resize_images():
    
    start = time()
    
    trainData = pd.read_pickle("train_data.pkl")
    
    data_directory = "train2014"
    
    output_directory = "train2014_normalized"
    
    image_filenames = list(trainData['file_name'][0:5])
    
    print(image_filenames)
    
    
    for f in image_filenames:
        
        filepath = os.path.join(data_directory, f)
        outpath = os.path.join(output_directory, f)
        
        #image = data.imread(filepath)
        
        #print(type(image))
        
        plt.imshow(image)
        
        result = transform.resize(image, (config.IMG_HEIGHT, config.IMG_WIDTH))
        
        #print(result.ndim)
        
        #plt.imshow(result)
        
        plt.imsave(outpath, result)
        
        #plt.imshow(data.imread(outpath))
    
    print("Standardization DONE IN", round(time() - start), "SEC")






def main():
	# Get num_tokens most frequent words in coco dataset (This will allow for a higher validation score)
	# Pull all the captions with those words
	# Tokenize words using nltk.tokenize
	# Pull all sentances within max_pad_length
	
	# Create word to index files based off these tokenized sentances, including pad, start, and stop tokens
	# Integrate padded files togeather into data_train and data_test pickles
	# Resize and clean images based on words in those files

	train_dict = build_vocab('train')
	write_captions('train', train_dict)


	print("temporary")



if __name__ == "__main__":
	main()







# Scrap function
def prepare_train_data(config):
    """ Prepare the data for training the model. """
    coco = COCO(config.train_caption_file)
    coco.filter_by_cap_len(config.max_caption_length)

    print("Building the vocabulary...")
    vocabulary = Vocabulary(config.vocabulary_size)
    if not os.path.exists(config.vocabulary_file):
        vocabulary.build(coco.all_captions())
        vocabulary.save(config.vocabulary_file)
    else:
        vocabulary.load(config.vocabulary_file)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))

    coco.filter_by_words(set(vocabulary.words))

    print("Processing the captions...")
    if not os.path.exists(config.temp_annotation_file):
        captions = [coco.anns[ann_id]['caption'] for ann_id in coco.anns]
        image_ids = [coco.anns[ann_id]['image_id'] for ann_id in coco.anns]
        image_files = [os.path.join(config.train_image_dir,
                                    coco.imgs[image_id]['file_name'])
                                    for image_id in image_ids]
        annotations = pd.DataFrame({'image_id': image_ids,
                                    'image_file': image_files,
                                    'caption': captions})
        annotations.to_csv(config.temp_annotation_file)
    else:
        annotations = pd.read_csv(config.temp_annotation_file)
        captions = annotations['caption'].values
        image_ids = annotations['image_id'].values
        image_files = annotations['image_file'].values

    if not os.path.exists(config.temp_data_file):
        word_idxs = []
        masks = []
        for caption in tqdm(captions):
            current_word_idxs_ = vocabulary.process_sentence(caption)
            current_num_words = len(current_word_idxs_)
            current_word_idxs = np.zeros(config.max_caption_length,
                                         dtype = np.int32)
            current_masks = np.zeros(config.max_caption_length)
            current_word_idxs[:current_num_words] = np.array(current_word_idxs_)
            current_masks[:current_num_words] = 1.0
            word_idxs.append(current_word_idxs)
            masks.append(current_masks)
        word_idxs = np.array(word_idxs)
        masks = np.array(masks)
        data = {'word_idxs': word_idxs, 'masks': masks}
        np.save(config.temp_data_file, data)
    else:
        data = np.load(config.temp_data_file).item()
        word_idxs = data['word_idxs']
        masks = data['masks']
    print("Captions processed.")
    print("Number of captions = %d" %(len(captions)))

    print("Building the dataset...")
    dataset = DataSet(image_ids,
                      image_files,
                      config.batch_size,
                      word_idxs,
                      masks,
                      True,
                      True)
    print("Dataset built.")
    return dataset