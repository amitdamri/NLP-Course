import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import time
from torch.utils.data import DataLoader


class TextClassificationModel(nn.Module):
    """
    Neural network class for text classification.
    """

    def __init__(self, vocab_size, embed_dim, layer1):
        """
        init the neural network architecture.
        :param vocab_size: the size of the vocav
        :param embed_dim: embedding dimension
        :param layer1: linear layer number of units
        """
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.seq = nn.Sequential(
            nn.Linear(embed_dim, layer1),
            nn.ReLU(),
            nn.Linear(layer1, 1),
        )

    def forward(self, text, offsets):
        """
        Feed the text into the model
        :param text: batch of text
        :param offsets: contains the size of each text
        :return: result
        """
        embedded = self.embedding(text, offsets)
        return self.seq(embedded)


class NeuralNetwork(object):
    """
    This class contains al the required function for training a TextClassificationModel.
    """

    def __init__(self):
        """
        Set the device to use.
        """
        self.device = torch.device("cpu")
        self.text_pipeline = None
        self.label_pipeline = None

    def get_pytorch_data(self, x, y):
        """
        Get data which can be used in pytorch models (label, text)
        :param x: data
        :param y: labels
        :return: list of tuples (label, text)
        """
        torch_data = []
        for index, row in enumerate(x['clean_text']):
            torch_data.append((y.iloc[index], row))
        return torch_data

    def init_vocabulary(self, torch_data):
        """
        Initialize the vocabulary using the given data
        :param torch_data: train data
        :return: vocabulary , english tokenizer
        """
        tokenizer = get_tokenizer('basic_english')

        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)

        vocab = build_vocab_from_iterator(yield_tokens(torch_data), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        return vocab, tokenizer

    def set_embedding(self, vocab):
        """
        Set the embedding weights and transform text tokens into vectors using the GloVe embedding
        :param vocab: vocabulary
        :return: the required embedding vectors (according to train set)
        """

        def unk_init(x):
            return torch.randn_like(x)

        embedding_glove = GloVe(name='6B', dim=300, unk_init=unk_init)
        pretrained_embedding = embedding_glove.get_vecs_by_tokens(vocab.get_itos())
        return pretrained_embedding

    def collate_batch(self, batch):
        """
        prepare batch data and set its device.
        :param batch:
        """
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)

    def train(self, dataloader, model, optimizer, criterion, epoch):
        """
        Run full epoch of training.
        :param dataloader: Dataloader object full with data
        :param model: text classification model
        :param optimizer: optimizer to use (Adam)
        :param criterion: criterion to use (Binary cross entropy with logits - apply sigmoid automatically)
        :param epoch: number of opech
        """
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()

        for idx, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)

            loss = criterion(predicted_label, label.float().unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (torch.round(torch.sigmoid(predicted_label)) == label.float().unsqueeze(1)).sum().item()
            total_count += label.float().unsqueeze(1).size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                                  total_acc / total_count))
                total_acc, total_count = 0, 0
                start_time = time.time()

    def evaluate(self, dataloader, model, criterion):
        """
        Evaluate the model on the given dataloader.
        :param dataloader: Dataloader object full with data
        :param model: model to evaluate
        :param criterion: criterion to use
        :return: accuracy, loss
        """
        model.eval()
        total_acc, total_count, loss = 0, 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = model(text, offsets)
                loss += criterion(predicted_label, label.float().unsqueeze(1))
                total_acc += (torch.round(torch.sigmoid(predicted_label)) == label.float().unsqueeze(1)).sum().item()
                total_count += label.float().unsqueeze(1).size(0)
        return total_acc / total_count, loss / total_count

    def fit(self, X_train, X_test, y_train, y_test, epochs=20, lr=0.01, batch_size=64):
        """
        The full training phase - using all the above functions.
            1. Prepare data for the pytorch model.
            2. Init the vocabulary.
            3. Create a TextClassification Neural network model,
            4. Set its embedding weights
            5. Set criterion
            6. Set optimizer
            7. Configure data loaders
            8. Train the model

        :param X_train: train data
        :param X_test: test data
        :param y_train: train labels
        :param y_test: test lables
        :param epochs: number of epochs to train
        :param lr: learning rate
        :param batch_size: the size of the batch
        :return: nn model, accuracy on test set
        """
        train_iter = self.get_pytorch_data(X_train, y_train)
        test_iter = self.get_pytorch_data(X_test, y_test)

        vocab, tokenizer = self.init_vocabulary(train_iter)

        self.text_pipeline = lambda x: vocab(tokenizer(x))
        self.label_pipeline = lambda x: int(x)

        vocab_size = len(vocab)
        emsize = 300
        model = TextClassificationModel(vocab_size, emsize, 50).to(self.device)
        model.embedding.weight.data = self.set_embedding(vocab)
        model.embedding.weight.requires_grad = False

        # Hyperparameters
        EPOCHS = epochs  # epoch
        LR = lr  # learning rate
        BATCH_SIZE = batch_size  # batch size for training

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)

        num_train = int(len(train_dataset) * 0.8)
        split_train_, split_valid_ = \
            random_split(train_dataset, [num_train, len(train_dataset) - num_train])

        train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                                      shuffle=True, collate_fn=self.collate_batch)

        valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                                      shuffle=True, collate_fn=self.collate_batch)

        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                     shuffle=True, collate_fn=self.collate_batch)

        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            self.train(train_dataloader, model, optimizer, criterion, epoch)
            accu_train, train_loss = self.evaluate(train_dataloader, model, criterion)
            accu_val, val_loss = self.evaluate(valid_dataloader, model, criterion)

            print('-' * 59)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                  'train accuracy {:8.3f} train loss {:8.3f} | valid accuracy {:8.3f} valid loss {:8.3f} '.format(epoch,
                                                                                                                  time.time() - epoch_start_time,
                                                                                                                  accu_train,
                                                                                                                  train_loss,
                                                                                                                  accu_val,
                                                                                                                  val_loss))
            print('-' * 59)

        test_accu, loss = self.evaluate(test_dataloader, model, criterion)
        print('-' * 59)
        print('| end of training | test accuracy {:8.3f}  '.format(test_accu))
        print('-' * 59)

        return model, test_accu
