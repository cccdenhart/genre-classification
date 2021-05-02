import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class QueryDataset(Dataset):
    def __init__(self, X, genre_labels):
        self.X = torch.tensor(X)
        self.unique_genres = set(genre_labels.tolist())
        self.genres = genre_labels

    def __len__(self):
        return len(self.genres)

    def __getitem__(self, idx):
        pos_genre = self.genres[idx]
        neg_genre = np.random.choice([g for g in self.unique_genres if g != pos_genre])
        return self.X[idx], pos_genre, neg_genrea


class QueryClassifier(pl.LightningModule):

    def __init__(self, input_shape, genre_embeddings, embedding_size=32, n_heads=4):
        super(QueryClassifier, self).__init__()
        self.genre_embeddings = genre_embeddings
        self.lstm = nn.LSTM(input_shape, embedding_size, 1, batch_first=True)
        # self.lstm2 = nn.LSTM(128, embedding_size, 1, batch_first=True)
        self.attention = nn.MultiheadAttention(embedding_size, n_heads)
        self.out = nn.Linear(embedding_size, 2)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.loss = nn.BCELoss()

    def forward(self, x, genres):
        lstm_out, _ = self.lstm(x)
        # lstm2_out, _ = self.lstm2(lstm1_out)
        b, l, e = lstm_out.shape
        genre_emb = torch.stack([self.genre_embeddings[int(g)] for g in genres])
        query = genre_emb.unsqueeze(0)
        key = lstm_out.reshape(l, b, e)
        attn_out, attn_weights = self.attention(query, key, key)
        out = self.sigmoid(self.activation(self.out(attn_out)))
        return out.squeeze(), attn_weights.detach()

    def shared_step(self, batch, step_type):
        x = batch[0]
        pos_genres = batch[1]
        neg_genres = batch[2]

        pos_out, _ = self(x, pos_genres)
        neg_out, _ = self(x, neg_genres)

        loss = (neg_out - pos_out).mean()  # TODO: check this

        pred = torch.cat([pos_out, neg_out])
        labels = torch.tensor([1] * len(pos_genres) + [0] * len(neg_genres))
        acc = (pred.argmax(-1) == labels).float().mean()

        self.log('%s/loss' % step_type, loss.item())
        self.log('%s/accuracy' % step_type, acc.item())

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def testing_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward(retain_graph=True)
