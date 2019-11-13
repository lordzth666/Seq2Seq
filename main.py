import torch
import torch.nn as nn
from torch import zeros, cat
import torch.nn.functional as F
from batch_loader import BatchLoader, tokenize
import numpy as np

WINDOW_SIZE = 35
# [a-z] [START] [EOS] [NULL]
NUM_CLASSES = 31

class RNNEncoder(nn.Module):
    def __init__(self, units, time_stamp=WINDOW_SIZE):
        super(RNNEncoder, self).__init__()
        self.W = nn.Linear(units*2, units)

    def forward(self, x):
        batch_dim = x.size(0)
        time_dim = x.size(1)
        out_dim = x.size(2)
        hidden_state = zeros([batch_dim, out_dim], dtype=torch.float)
        for i in range(WINDOW_SIZE):
            x_in = cat([x[:, i, :], hidden_state], dim=1)
            out = self.W(x_in)
            out = F.relu(out)
            hidden_state = out
        return out

class OneShotRNNDecoder(nn.Module):
    def __init__(self, units, num_classes=2):
        super(OneShotRNNDecoder, self).__init__()
        self.W = nn.Linear(units, num_classes) 

    def forward(self, x):
        return self.W(x)


class Seq2SeqModel(nn.Module):
    def __init__(self, units, num_classes=31, window_size=WINDOW_SIZE):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Linear(num_classes, units)
        self.encoder = RNNEncoder(units, window_size)
        self.decoder = OneShotRNNDecoder(units)

    def forward(self, x):
        return self.decoder(self.encoder(self.embedding(x)))

def trainval(model, data_loader):
    EPOCHS = 30
    train_batch_size = 64
    test_batch_size = 64

    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, eps=1e-8, momentum=0.9, weight_decay=1e-4)

    global_steps = 0

    for i in range(EPOCHS):
        train_steps = data_loader.num_train_examples // train_batch_size
        test_steps = data_loader.num_test_examples // test_batch_size

        for j in range(train_steps):
            inputs, targets = data_loader.get_train_batch(batch_size=train_batch_size)
            inputs, targets = torch.from_numpy(inputs), torch.from_numpy(targets)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Update the weights
            loss.backward()
            optimizer.step()
            # Increase global step and print out log
            global_steps += 1
            if global_steps % 50 == 0:
                bin_acc = outputs.argmax(1).eq(targets).sum().item()
                bin_acc = float(bin_acc) / float(outputs.size(0))
                print("[Step=%d]\tLoss=%f\tAccuracy=%f" %(global_steps, loss.item(), bin_acc))
            pass
        
        avg_loss = 0
        avg_acc = 0

        print("Validation...")

        with torch.no_grad():
            for j in range(test_steps):
                inputs, targets = data_loader.get_test_batch(batch_size=train_batch_size, id=j)
                inputs, targets = torch.from_numpy(inputs), torch.from_numpy(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                bin_acc = outputs.argmax(1).eq(targets).sum().item()
                bin_acc = float(bin_acc) / float(outputs.size(0))
                avg_acc += bin_acc / test_steps
                avg_loss += loss / test_steps
                pass
        print("Validation Loss=%f\tValidation Accuracy=%f" %(avg_loss, avg_acc))
    pass

def inference(model, inputs, targets):
    with torch.no_grad():
        tokenized_inputs = tokenize(inputs, window_size=WINDOW_SIZE)
        tokenized_inputs = np.asarray(tokenized_inputs, dtype=np.float32)
        tokenized_inputs = np.expand_dims(tokenized_inputs, axis=0)
        tokenized_inputs = torch.from_numpy(tokenized_inputs)

        outputs = model(tokenized_inputs).cpu().numpy()

    print("Target is: [%s]" %targets)

    if outputs[0][0] < outputs[0][1]:
        print("Actual output is: [polo ]")
    else:
        print("Actual output is: [ ]")
    pass


def main():
    np.random.seed(233)
    torch.manual_seed(233)
    data_loader = BatchLoader(10000)
    model = Seq2SeqModel(10, NUM_CLASSES, WINDOW_SIZE)
    trainval(model, data_loader)

    # Should print 'polo '
    inference(model, "nor marco I", "polo ")
    inference(model, "marco nor I", "polo ")
    inference(model, "nor I marco", "polo ")
    
    # Should print ' '
    inference(model, "nor I neither", " ")

    # More difficult task
    inference(model, "nor ma rco I", " ")
    inference(model, "ma rco nor I", " ")

    pass

if __name__ == "__main__":
    main()
