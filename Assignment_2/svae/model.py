import torch
from torch import nn
import numpy as np
from collections import defaultdict
import time

from .Base_model import BaseModel


class SVAEModel(nn.Module):
    def __init__(self, opt, vocab_size):
        """Initialize the Sentence VAE Language Model class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(SVAEModel, self).__init__()

        self.opt = opt
        self.vocab_size = vocab_size
        self.input_size = opt.input_size
        self.hidden_size = opt.hidden_size
        self.num_layers = opt.num_layers
        self.latent_size = opt.latent_size

        self.batch_size = opt.batch_size if opt.mode == 'train' else opt.test_batch

        # Set the RNN model structure
        self.word_embeddings = nn.Embedding(self.vocab_size, self.input_size)

        self.encoder = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.decoder = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        hidden_factor = self.hidden_size * self.num_layers
        self.hidden2mean = nn.Linear(in_features=hidden_factor, out_features=self.latent_size)
        self.hidden2logv = nn.Linear(in_features=hidden_factor, out_features=self.latent_size)
        self.latent2hidden = nn.Linear(in_features=self.latent_size, out_features=hidden_factor)

        self.outputs2vocab = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

        # Initialize the weights
        self.init_weights()

    def set_input(self, input):
        """load input data from the dataloader.

        Parameters:
            input: includes the input data.
        """
        pass

    def init_weights(self):
        init_range = 0.1
        self.word_embeddings.weight.data.uniform_(-init_range, init_range)
        self.hidden2mean.weight.data.uniform_(-init_range, init_range)
        self.hidden2mean.bias.data.zero_()
        self.hidden2logv.weight.data.uniform_(-init_range, init_range)
        self.hidden2logv.bias.data.zero_()
        self.latent2hidden.weight.data.uniform_(-init_range, init_range)
        self.latent2hidden.bias.data.zero_()
        self.outputs2vocab.weight.data.uniform_(-init_range, init_range)
        self.outputs2vocab.bias.data.zero_()

    def encode(self, input, batch_size):
        embeddings = self.word_embeddings(input)
        _, hidden = self.encoder(embeddings)
        hidden = hidden.view(batch_size, self.hidden_size * self.num_layers)

        # Reparameterization trick
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = self.softplus(torch.exp(0.5 * logv))

        return embeddings, logv, mean, std

    def decode(self, embeddings, mean, std, batch_size, num_samples):
        pred = torch.Tensor().cuda() if torch.cuda.is_available() else torch.Tensor()
        for s in range(num_samples):
            # Generate the latent space
            eps = torch.randn([batch_size, self.latent_size])
            if torch.cuda.is_available():
                eps = eps.cuda()
            z = eps * std + mean
            hidden = self.tanh(self.latent2hidden(z))
            hidden = hidden.view(self.num_layers, batch_size, self.hidden_size)

            # decoder
            output, _ = self.decoder(embeddings, hidden)
            logp = self.outputs2vocab(output)
            logp = logp.view(batch_size, -1, self.vocab_size, 1)
            pred = torch.cat((pred, logp), dim=3)

        pred = torch.mean(pred, dim=3)

        return pred, z

    def inference(self, z, n, seq_length, pad_idx, sos_idx, eos_idx, method):
        z = torch.randn([n, self.latent_size]) if z is None else z

        hidden = self.tanh(self.latent2hidden(z))
        hidden = hidden.view(self.num_layers, n, self.hidden_size)

        sequence_idx = torch.arange(0, n).long()
        sequence_running = torch.arange(0, n).long()
        sequence_mask = torch.ones(n).byte()
        running_seqs = torch.arange(0, n).long()
        generations = torch.Tensor(n, seq_length).fill_(pad_idx).long()

        if torch.cuda.is_available():
            sequence_idx = sequence_idx.cuda()
            sequence_running = sequence_running.cuda()
            sequence_mask = sequence_mask.cuda()
            running_seqs = running_seqs.cuda()
            generations = generations.cuda()

        t = 0
        while (t < seq_length and len(running_seqs) > 0):
            if t == 0:
                input_sequence = torch.Tensor(n).fill_(sos_idx).long()
                if torch.cuda.is_available():
                    input_sequence = input_sequence.cuda()
            if len(running_seqs) == 1 and len(input_sequence.size()) == 0:
                input_sequence = input_sequence.unsqueeze(0)
            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.word_embeddings(input_sequence)
            output, hidden = self.decoder(input_embedding, hidden)
            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits, method=method)
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            sequence_mask[sequence_running] = (input_sequence != eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            running_mask = (input_sequence != eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            if len(running_seqs) > 0:
                if len(running_seqs) == 1 and len(input_sequence.size()) == 0:
                    pass
                else:
                    input_sequence = input_sequence[running_seqs]
                    hidden = hidden[:, running_seqs]
                running_seqs = torch.arange(0, len(running_seqs)).long()
                if torch.cuda.is_available():
                    running_seqs = running_seqs.cuda()

            t += 1

        return generations, z

    def _sample(self, dist, method='greedy'):
        if method == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        elif method == 'multi':
            batches = dist.size(0)
            sample = torch.randn(batches, 1).cuda().long() if torch.cuda.is_available() else torch.Tensor(batches, 1).long()
            for k in range(batches):
                word_weights = dist[k].squeeze().exp()
                sample[k] = torch.multinomial(word_weights, 1)[0]

        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        running_latest = save_to[running_seqs]
        running_latest[:, t] = sample.data
        save_to[running_seqs] = running_latest

        return save_to
