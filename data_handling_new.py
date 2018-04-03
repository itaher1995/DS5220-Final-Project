
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

#import nltk
#nltk.download('punkt')

def get_data(file):

	data = pd.read_csv(file)

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

def make_word_idx_files(dictionary):    
    
    idx_to_word = {}
    for i in range(len(dictionary)):
        idx_to_word[i+1] = dictionary[i]
    
    ## Include start word and stop word
    idx_to_word[config.START_TOKEN_IDX] = "_start_"
    idx_to_word[config.STOP_TOKEN_IDX] = "_stop_"


    word_to_idx = {word: idx for idx, word in idx_to_word.items()}

    print(idx_to_word)
    print(word_to_idx)

    # Writes the tokens to token file with index
    with open('idx_to_word.pkl', 'wb') as f:
        pickle.dump(idx_to_word, f)
    with open('word_to_idx.pkl', 'wb') as f:
        pickle.dump(word_to_idx, f)

def padder(sentance):
    
    sentance = [config.START_TOKEN_IDX] + sentance + [config.STOP_TOKEN_IDX]
    print(sentance)
    
    # Want to pad up to max length
    num_pad = config.MAX_CAP_LEN + 3 - len(sentance)

    # Pads strings with zero's. Accounted for this when we mapped the word idx's starting at 1
    padded_sentance = np.pad(caption_array, (0,num_pad), 'constant', constant_values = (0,0))

    return padded_sentance

def get_captions(filetype,dictionary):
    """
    Get's captions that only contain words in dictionary
    """

    filename = filetype + "_annotations.csv"
    data = get_data(filename)
    
    captions = pd.DataFrame()

    for idx, row in data.iterrows():
        ann_words = word_tokenize(row['caption'])

        # Makes sure we're only getting rows within the size range we want
        if len(ann_words) < config.MAX_CAP_LEN:
            #print(5)
            continue

        #print(1)
        for word in ann_words:
            #print(2)
            if word not in dictionary:
                #print(3)
                continue
        #print(3)
        captions = pd.concat([captions, row], 1)

    captions = captions.T
    #print(len(captions))
    #print(captions.head())

    return captions

def attach_annotations(imgData, annData):

    start = time()
    condencedData = pd.DataFrame()

    word_to_idx = pd.read_pickle("word_to_idx.pkl")

    # Cycles through each image in the image data table
    for img in imgData['image_id']:
        
        # Returns the subset of annotations with a particular image_id
        annSub = annData[annData['image_id'] == img]
        annDict = {}
        
        matrix = []

        # Goes through each row of annotation subset for a specific image id and makes a single dictionary
        # where the key is the caption_id and the value is the caption
        for idx, row in annSub.iterrows():
            annDict[row['caption_id']] = row['caption']
            
            # Gets the tokenized sequence of indexes corresponding to words
            idx_annotation = [word_to_idx[word] for word in word_tokenize(row['caption'])]
            idx_annotation = padder(idx_annotation)
            matrix.append(idx_annotation)
        
        # Makes a pd.Series to updata a new df with the image_id and associated annotations
        entry = pd.Series()
        entry['image_id'] = img
        entry['annotations'] = annDict
        entry['idx_caption_matrix'] = matrix
        
        # Adds column into df
        condencedData = pd.concat([condencedData, entry], 1)
    
    condencedData = condencedData.T
    
    # Merges the new df of image_id's and annotations into image data table
    imgData = imgData.merge(condencedData, on = 'image_id', how = 'left')
    
    print("DONE IN", round(time() - start), "SEC")
    
    return imgData


def merge_img_caption_data(captions, filetype):
    
    file = "img_" + filetype + "image_data.csv"
    
    data = pd.read_csv(file)
    data.rename(columns = {'id':'image_id'})

    captions.rename(columns = {'id':'caption_id'})
    
    image_ids = captions['image_id'].copy()

    # data_subset is images who have corresponding captions
    data_subset = data[data['image_id'] == captions['image_id']]

    joined_data = attach_annotations(data_subset, captions)

    outfile = filetype + "_data.pkl"

    joined_data.to_pickle(outfile)


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
	
    # Pull all the captions with those words, include max_pad_length operation here
	# Tokenize words using nltk.tokenize
	
	# Create word to index files based off these tokenized sentances, including pad, start, and stop tokens
	# Integrate padded files togeather into data_train and data_test pickles
	# Resize and clean images based on words in those files

    # Get num_tokens most frequent words in coco dataset (This will allow for a higher validation score)
    train_dict = build_vocab('train')

    # Create word to index files based off these tokenized sentances, including pad(0), start (len_dict+1), and stop tokens (len_dict+2)
    #make_word_idx_files(train_dict)

    # Integrates startwords, stopwords, and padding to captions who's words are in the dictionary
    train_caps = get_captions('train', train_dict)
    val_caps = get_captions('val', train_dict)

    print(len(train_caps))
    print(train_caps.head())

    print(len(val_caps))
    print(val_caps.head())
    #merge_img_caption_data(train_caps, 'train')
    #merge_img_caption_data(val_caps, 'val')


    
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