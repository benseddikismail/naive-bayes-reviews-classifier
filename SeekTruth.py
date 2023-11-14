import sys
import re 
import numpy as np

def load_file(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed)>0 else "")
            objects.append(parsed[1] if len(parsed)>1 else "")

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}

def preprocess(data):
    
    stop_words = [
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he',
        'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
        'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'were', 'be',
        'been', 'being', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
        'between', 'into', 'through', 'during', 'before', 'after', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        'can', 'will', 'just', 'should', 'now', 'us', 'however', 'yet', 'still', 'never', 'every', 'need', "won't", 'must',
        "it's", "don't", 'none', 'could', 'against', 'though', 'although', 'anything', 'anymore', 'absolutely', 'sure', 'would', 'like', 'etc',
        'across', 'according', 'among', 'less', 'nothing', 'strongly', 'show', 'try', 'highly', 'unfortunately', 'unless'
    ]
    clean_data = []
    for review in data:
        lowercase_data = review.lower()
        tmp = ' '.join([word for word in lowercase_data.split() if word not in stop_words])
        data_without_punctuation = re.sub(r'[^\w\s]', '', tmp)
        clean_data.append(data_without_punctuation)
    return clean_data

# bag of words
def get_bag_of_words(data):
    #lowercase_data = data.lower()
    #data_without_punctuation = re.sub(r'[^\w\s]', '', lowercase_data)
    bag = data.split()
    #cleaned_bag = [word for word in bag if word not in stop_words]
    #cleaned_bag = list(set(bag)) # remove duplicates
    return bag

def word_frequencies(bag_of_words, reviews, labels):

    word_freq_dict = {
        # word1: [0,1,2,0,1,...]
        # word2: [0,1,0,1,0,...]
    }

    truthful_freq_dict, deceptive_freq_dict = {}, {}

    for review in bag_of_words:
        for word in review:
            word_freq_dict[word] = [0] * len(reviews)
            truthful_freq_dict[word] = [0] * len(reviews)
            deceptive_freq_dict[word] = [0] * len(reviews)

    for i, review in enumerate(reviews):
        res = review.split() 
        for word in res:
            word_freq_dict[word][i] += 1
            if labels[i] == "truthful":
                truthful_freq_dict[word][i] += 1
            else:
                deceptive_freq_dict[word][i] += 1

    return (truthful_freq_dict, deceptive_freq_dict, word_freq_dict)

# classifier : Train and apply a bayes net classifier
#
# This function should take a train_data dictionary that has three entries:
#        train_data["objects"] is a list of strings corresponding to reviews
#        train_data["labels"] is a list of strings corresponding to ground truth labels for each review
#        train_data["classes"] is the list of possible class names (always two)
#
# and a test_data dictionary that has objects and classes entries in the same format as above. It
# should return a list of the same length as test_data["objects"], where the i-th element of the result
# list is the estimated classlabel for test_data["objects"][i]
#
# Do not change the return type or parameters of this function!
#
def classifier(train_data, test_data):

    train_reviews, train_labels, test_reviews = preprocess(train_data["objects"]), train_data["labels"], preprocess(test_data["objects"])

    bag_of_words = list()
    truthful_reviews, deceptive_reviews = [], []
    N_truthful_words, N_deceptive_words = 0, 0 # number of words per class
    N_unique_words = 0 # number of words in the bag of words
    for i in range(len(train_reviews)):
        bag_of_words.append(get_bag_of_words(train_reviews[i]))
        N_unique_words += len(bag_of_words[i])
        if train_labels[i] == "truthful":
            truthful_reviews.append(train_reviews[i])
            N_truthful_words += len(train_reviews[i].split())
        else:
            deceptive_reviews.append(train_reviews[i])
            N_deceptive_words += len(train_reviews[i].split())

    # word frequencies per review
    truthful_freq_dict, deceptive_freq_dict, word_frequencies_dict = word_frequencies(bag_of_words, train_reviews, train_labels)

    # class prior probabilities P(truthful) and P(deceptive)
    data_len = len(train_reviews)
    p_truthful = float(len(truthful_reviews)/data_len)
    p_deceptive = float(len(deceptive_reviews)/data_len)

    alpha = 0.3 # Laplace smoothing

    # calculate likelihoods
    likelihoodTable_truthful = {word: 0 for word in word_frequencies_dict}
    likelihoodTable_deceptive = {word: 0 for word in word_frequencies_dict}
    for word in word_frequencies_dict:
        word_freq_truthful = sum(truthful_freq_dict[word]) # number of occurences of word in truthful reviews
        p_word_given_truthful = (word_freq_truthful + alpha) / (N_truthful_words + alpha*N_unique_words)
        likelihoodTable_truthful[word] = p_word_given_truthful
        # --
        word_freq_deceptive = sum(deceptive_freq_dict[word]) # number of occurences of word in truthful reviews
        p_word_given_deceptive = (word_freq_deceptive + alpha) / (N_deceptive_words + alpha*N_unique_words)
        likelihoodTable_deceptive[word] = p_word_given_deceptive

    predictions = []
    # classify test reviews
    for review in test_reviews:
        words = review.split()
        # using log to avoid underflow stemming from multiplying small probabilities
        p_truthful_given_review = np.log(p_truthful)
        p_deceptive_given_review = np.log(p_deceptive)
        for word in words:
            if word in likelihoodTable_truthful:
                p_truthful_given_review += np.log(likelihoodTable_truthful[word])
            if word in likelihoodTable_deceptive:
                p_deceptive_given_review += np.log(likelihoodTable_deceptive[word]) 
                         
        if p_deceptive_given_review > p_truthful_given_review:
            predictions.append("deceptive")
        else:
            predictions.append("truthful")
    
    return predictions


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv
    # Load in the training and test datasets. The file format is simple: one object
    # per line, the first word one the line is the label.
    train_data = load_file(train_file)
    test_data = load_file(test_file)
    if(sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    # make a copy of the test data without the correct labels, so the classifier can't cheat!
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}

    results = classifier(train_data, test_data_sanitized)

    # calculate accuracy
    correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))
    
# References
# https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece