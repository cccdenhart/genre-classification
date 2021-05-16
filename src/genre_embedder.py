import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class MultiClassDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GenreEmbedder(pl.LightningModule):

	def __init__(self, input_dim, n_class, embedding_size=32):
		super().__init__()

		self.cnn = nn.Sequential(
			nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
			# nn.BatchNorm2d(1),
			# nn.ReLU(inplace=True),
			# nn.MaxPool2d(kernel_size=2, stride=1),
		)
		self.in_layer = nn.Linear(52, 48)
		self.embedding = nn.Linear(52, embedding_size)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(0.1)
		self.out = nn.Linear(embedding_size, n_class)

		self.loss = nn.CrossEntropyLoss()

	def forward(self, x):
		if len(x.shape) > 2:
			x = x.squeeze()

		b, f = x.shape
		x_reshape = x.reshape(b, 1, 4, 13)
		conv_layer = self.activation(self.cnn(x_reshape)).reshape(b, 52)
		embedding = self.activation(self.embedding(conv_layer))
		out = self.out(embedding)
		return out, embedding

	def shared_step(self, batch, step_type):
		x, y = batch
		y_hat, _ = self(x)
		loss = self.loss(y_hat, y)
		self.log(f'{step_type} loss', loss.item())
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
