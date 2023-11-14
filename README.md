# Naive Bayes Classifier
Naive Bayes classifier of reviews into Truthful/Deceptive.
## Data Pre-processing
The quality of the data has a significant impact on the perfomance of the classifier. Hence, data is cleaned before its usage by bringing all words to lower case, removing punctuation, and ignoring stop words. A list of stop words is tuned as a hyperparameter as the words it encompasses affect the accuracy of the classifier. The stop words were carefully selected and retained in a way that they don't carry semantic information, in order to prevent information loss and underfitting.
## Naive Bayes Classifier
To classify the reviews as ***truthful*** or ***deceptive***, the following probabilties ought to be calculated in accordance with Bayes rule:  

$$ P(truthful|w_1, w_2, ..., w_n) \propto P(truthful)\prod_{i=1}^{n} \cdot P(w_i|truthful) $$

$$ P(deceptive|w_1, w_2, ..., w_n) \propto P(deceptive)\prod_{i=1}^{n} \cdot P(w_i|deceptive) $$

> $w_1, w_2, ..., w_n$ are words in the reviews. Note that not dividing by the marginal probability of the words is due to the Naive Bayes classifier's assumption that the presence of a word in a class is independent of the presence of other words.

Prior probabilities $P(truthful)$ and $P(deceptive)$, are computed by dividing the numbers of truthful reviews and deceptive reviews, respectively, by the total number of reviews.  
A bag of words is used as as a set all pre-processed words. Based on it, the frequencies of each word in the reviews constitute likelihood tables which are maintained to simplify the process of computing the conditional probabilities.  
Likelihoods, are computed as:  

$$ P(w_i|truthful) = \frac{ \text{frequency of } w_i \text{ in truthful reviews} + \alpha}{\text{total number of words in truthful reviews} + \alpha \times \text{total number of words in the bag of words}} $$

$$ P(w_i|deceptive) = \frac{ \text{frequency of } w_i \text{ in deceptive reviews} + \alpha}{\text{total number of words in deceptive reviews} + \alpha \times \text{total number of words in the bag of words}} $$

&rarr; $\alpha$ is the Laplace smoothing parameter. It adresses the problem of having zero probabilities in case a word in the test data is new to the classifier. $\alpha$ ensures that all words have non-zero probabilities of occuring in a class. It is also a hyperparameter that was tweaked to a value of 0.3 for an optimal performance of the classifier. Generally, the smaller the value of the smoothing parameter the more the classifier is regularized and the impact of smoothing on the original data is minimal. 
## Classifier Accuracy and Future Work
The accuracy of the classifier is **87.75%**. Since its performance is directly influenced by data cleaning, future work would mostly cover further pre-processing:
- Compile a more comprehensive list of stop words that minimizes information loss and maximizes the perfomance of the model.
- Analyze the data even more to perform feature engineering and, for example, only keep the most relevant features (words).
- Normalize words through stemming/lemmatization using specialized libraries like NLTK.
- Assemble more training examples.
- Clean and optimize the code.