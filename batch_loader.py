import numpy as np
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from tqdm import tqdm 

def create_onehot_vector(idx, num_classes=31):
    one_hot_vector = np.zeros(num_classes)
    one_hot_vector[idx] = 1
    return one_hot_vector

def tokenize(rstr, window_size=35):
    # 0: <START> 1-27: [a-z] 28: ' ' 29: <EOS> 30: <NULL>
    tokens = []
    tokens.append(create_onehot_vector(0))
    rstr_lower = rstr.lower()
    for x in rstr_lower:
        if 'a' <= x and x <= 'z':
            tokens.append(create_onehot_vector(ord(x)-96))
        if x == ' ':
            tokens.append(create_onehot_vector(28))
    tokens.append(create_onehot_vector(29))
    while np.shape(tokens)[0] < window_size:
        tokens.append(create_onehot_vector(30))
    tokens = np.asarray(tokens)
    return tokens
            
def generate_batch_data(words, clipping_window_size=35):
    # generate number of words for a string
    num_words = np.random.randint(2, 6)
    rand_idx = np.random.choice(np.arange(np.shape(words)[0]), num_words)
    words_ = words[rand_idx]
    # randomly insert 'marco': Data Augmentation
    p = np.random.rand()
    if p < 0.5:
        words_ = np.append(words_, 'marco')
    elif p < 0.6:
        words_ = np.append(words_, 'mar')
        words_ = np.append(words_, 'co')
    elif p < 0.7:
        words_ = np.append(words_, 'ma')
        words_ = np.append(words_, 'rco')
    np.random.shuffle(words_)
    rstr = ' '.join(words_)
    rstr = rstr[:clipping_window_size-2]
    data = tokenize(rstr, clipping_window_size)
    if (rstr.find('marco') != -1):
        label = 1
    else:
        label = 0
    return data, label

class BatchLoader:
    def __init__(self, num_data):
        self.words = []
        print("Preprocessing batch loader...")
        for word in brown.words():
            if not word in [".", "\t", " ", "?", "!", '*', "\'", "\"", ","]:
                self.words.append(word.strip('\''))
        self.words = np.asarray(self.words)
        print("Preprocessing finished!")
        data_examples = num_data

        data_X = []
        data_y = []
        print("Generating data...")
        for i in tqdm(range(data_examples)):
            tokens, label = generate_batch_data(self.words)
            data_X.append(tokens)
            data_y.append(label)

        data_X = np.asarray(data_X, dtype=np.float32)
        data_y = np.asarray(data_y, dtype=np.long)

        print(data_X.shape)
        print(data_y.shape)

        print(data_y)

        self.data_train_X, self.data_test_X, self.data_train_y, self.data_test_y = train_test_split(data_X, data_y, test_size=0.20)
        self.num_train_examples = np.shape(self.data_train_X)[0]
        self.num_test_examples = np.shape(self.data_test_X)[0]

    def get_train_batch(self, batch_size=32):
        batch_id = np.random.choice(np.arange(self.num_train_examples), batch_size)
        return self.data_train_X[batch_id, :], self.data_train_y[batch_id]

    def get_test_batch(self, batch_size=32, id=0):
        batch_id = np.arange(id*batch_size, id*batch_size+batch_size)
        return self.data_test_X[batch_id, :], self.data_test_y[batch_id]


        

    
    
        
    
    
