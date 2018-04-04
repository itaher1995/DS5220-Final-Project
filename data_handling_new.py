


"""
Generally, the code runs pretty well and quickly until the section where it starts
to merge the two files togeather. I imagine what is slowing everything down in particular
is the subset_by_image_id on line 213, but I could think of a way to quickly subset
the image data id's by the caption data id's when the other attempt failed. Something
else could be slowing things down too. Ooooo or the same pd.concat trick I was trying
to use earlier in line 199. I bet that's it actually. All we want is to add a dict 
of captions ids + english captions and a matrix of tokenized captions as columns to 
the image data files
"""


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

# Need to install at least once for the word_tokenize, or at least that worked for me
#import nltk
#nltk.download('punkt')

def clean_string(string):
    #string = string.lower().strip('.')
    string = word_tokenize(string.lower())
    string = " ".join(string)
    
    return string


# Basic idea is to return unique instances from all the words for all captions in data set as vocab
def get_tokens(data):
    captions = []
    for idx, row in data.iterrows():
        value = row['caption']
        #captions.append(word_tokenize(value))
        captions.append(clean_string(value))
    joined_cap = " ".join(captions) # join all the captions into one string
    tokenized_cap = joined_cap.split() # Split words individually
    #unique_tokens = sorted(list(set(tokenized_cap)))

    return tokenized_cap


# Returns the top config.NUM_TOKENS words in terms of frequency in the dataset
def build_vocab(filetype):
    filename = filetype + "_annotations.csv"
    data = pd.read_csv(filename)

    words = get_tokens(data)

    # Counts the number of words
    word_counts = Counter(words)

    print(len(word_counts))

    top_words = word_counts.most_common(config.NUM_TOKENS)

    words_for_dict = {}
    
    for word, count in top_words:
        words_for_dict[word] = count

    print(len(words_for_dict))
    print(words_for_dict)
    return words_for_dict


# Create the indexed word files, including spaces for padding, start word, and stop word
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


# Pads captions with 0's and adds start word at beginning of token and stop word at end
def padder(sentance):
    
    sentance = [config.START_TOKEN_IDX] + sentance + [config.STOP_TOKEN_IDX]
    print(sentance)
    
    # Want to pad up to max length, and need to account for start and stop words
    num_pad = config.MAX_CAP_LEN + 2 - len(sentance)

    # Pads strings with zero's. Accounted for this when we mapped the word idx's starting at 1
    padded_sentance = np.pad(caption_array, (0,num_pad), 'constant', constant_values = (0,0))

    return padded_sentance


# Filters the dataset to get all the captions where the words are contained in the dictionary 
# Also filters to get captions below the max_cap_length
def get_captions(filetype,dictionary):
    """
    Get's captions that only contain words in dictionary
    """
    start = time()

    filename = filetype + "_annotations.csv"
    data = pd.read_csv(filename)
    
    idx_list = []

    for idx, row in data.iterrows():
        #ann_words = word_tokenize(row['caption'].lower())
        ann_words = clean_string(row['caption']).split()

        # Makes sure we're only getting rows within the size range we want
        if len(ann_words) > config.MAX_CAP_LEN:
            #print(5)
            continue

        in_dict = True
        #print(in_dict)
        #print(1)
        for word in ann_words:
            #print(2)
            if word not in dictionary:
                #print(3)
                in_dict = False
                break
        #print(3)
        #print(in_dict)
        if in_dict == True:
            idx_list.append(idx)

        # Keeps track of number of captions during process
        if not len(idx_list) % 10000:
            print("Number of captions:",len(idx_list), "   Current Time:",round(time() - start))

    data = data.loc[idx_list,:]
    #print(len(captions))
    #print(captions.head())

    print("DONE IN", round(time() - start), "SEC")

    return data


# Attaches the word annotations and indexed (numerical) annotations to the image data
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
            ann = row['caption']
            
            # Gets the tokenized sequence of indexes corresponding to words
            idx_annotation = [word_to_idx[word] for word in word_tokenize(row['caption'])]
            idx_annotation = padder(idx_annotation)
            matrix.append(idx_annotation)

        # Makes a pd.Series to updata a new df with the image_id and associated annotations
        entry = pd.Series()
        entry['image_id'] = img
        entry['annotations'] = ann
        entry['idx_caption_matrix'] = matrix

        # Adds column into df
        condencedData = pd.concat([condencedData, entry], 1)
    
    condencedData = condencedData.T
    print(condencedData.head())
    
    # Merges the new df of image_id's and annotations into image data table
    imgData = imgData.merge(condencedData, on = 'image_id', how = 'left')
    
    print("DONE IN", round(time() - start), "SEC")
    
    return imgData


# Get the subset of the image data by the image id's in the filtered captions
def subset_by_image_id(data, captions):
    
    new_data = pd.DataFrame()

    ids = list(set(captions['image_id']))

    # Loops through the unique id's of the filtered captions dataset and combines
    # the subsets of the image data per image_id
    for i in ids:
        temp = data[data['image_id'] == i].copy()
        new_data = pd.concat([new_data, temp])

    return new_data


# Merges the image data and caption data togeather in a pickle file
# Goal is to have the image_id's, caption_id's, image_filepath, captions,
# and tokenized captions all in the same pkl file
def merge_img_caption_data(captions, filetype):
    
    file = filetype + "_image_data.csv"
    
    data = pd.read_csv(file)
    data = data.rename(columns = {'id':'image_id'})

    captions = captions.rename(columns = {'id':'caption_id'})

    # data_subset is images who have corresponding captions
    # Really all this is supposed to do is makes sure all of the
    # captions have id's and all the images have captions
    data_subset = subset_by_image_id(data,captions)

    joined_data = attach_annotations(data_subset, captions)
    print(joined_data.head())

    outfile = filetype + "_data.pkl"
    print(outfile)

    #joined_data.to_pickle(outfile)


# Resizes images based on the image data we've filtered
def resize_images(filetype):
    
    start = time()
    
    datapath = filetype + "_data.pkl"

    data = pd.read_pickle(datapath)
    
    data_directory = filetype + "2014"
    
    output_directory = data_directory + "_normalized"
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Only prints first 5 here for testing purposes
    image_filenames = list(data['file_name'][0:5])
    
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
    train_dict = build_vocab('train')

    # Create word to index files based off these tokenized sentances, including pad(0), start (len_dict+1), and stop tokens (len_dict+2)
    # Already done so currently commeneted out
    #make_word_idx_files(train_dict)

    # Integrates startwords, stopwords, and padding to captions who's words are in the dictionary
    train_caps = get_captions('train', train_dict)
    val_caps = get_captions('val', train_dict)

    print(len(train_caps))
    print(train_caps.head())

    print(len(val_caps))
    print(val_caps.head())
    
    # Merge the captions with the image data
    merge_img_caption_data(train_caps, 'train')
    merge_img_caption_data(val_caps, 'val')

    # Resize the image files that correspond to the image_id's of the filtered captions
    # Last step once everything else is complete
    #resize_images("train")
    #resize_images("val")

    # Needs to be done seperately, will require different logic
    #resize_images("test")

    
    print("temporary")


if __name__ == "__main__":
	main()
