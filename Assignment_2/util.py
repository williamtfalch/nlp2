import os
import sys
import numpy as np
import torch
from torch import nn
import time

def kl_weight_function(anneal, step, k=0.0025, x0=2500):
    if anneal == 'Logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal == 'Linear':
        return min(1, step / x0)
    else:
        assert False, 'Wrong KL annealing function'

def train_model(model, dataset, epoch, lr, opt):
    print("-----------------------------------Training-----------------------------------")
    model.train()

    # Ignore the padding tokens for the loss computation
    pad_index = dataset.vocabulary.word2token['-PAD-']
    criterion_loss = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')

    vocab_size = len(dataset.vocabulary)
    data_size = len(dataset.train_data)
    start = time.time()
    total_loss = []
    numerator = 0.0
    denominator = 0.0
    perplexity = 0.0
    accuracy = []

    for batch, idx in enumerate(range(0, data_size - 1, opt.batch_size)):
        source, target, sentence_len = dataset.load_data('train', idx, opt.batch_size)
        if source is None:
            continue

        if torch.cuda.is_available():
            source = source.cuda()
            target = target.cuda()

        model.zero_grad()
        embeddings, logv, mean, std = model.encode(source, opt.batch_size)
        output, _ = model.decode(embeddings, mean, std, opt.batch_size, num_samples=opt.sample_size)
        output = output.view(opt.batch_size * opt.seq_length, vocab_size)
        target = target.view(opt.batch_size * opt.seq_length)
        NLL_loss = criterion_loss(output, target)
        # Get the KL loss term and the weight
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_weight_function(anneal=opt.anneal, step=batch)
        loss = (NLL_loss + KL_weight * KL_loss)
        loss.backward(retain_graph=True)
        total_loss.append(loss.cpu().item() / (opt.batch_size * opt.seq_length))

        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        # Compute the perplexity
        numerator += loss.cpu().item()
        denominator += np.sum(sentence_len)
        perplexity = np.exp(numerator / denominator)

        # Compute the word prediction accuracy
        output = output.view(opt.batch_size, -1, vocab_size)
        target = target.view(opt.batch_size, -1)
        acc = compute_accuracy(output, target, sentence_len, pad_index)
        accuracy.append(acc)

        if (batch % opt.print_interval == 0) and batch != 0:
            elapsed_time = (time.time() - start) * 1000 / opt.print_interval
            print('Epoch: {:5d} | {:5d}/{:5d} batches | LR: {:5.4f} | loss: {:5.4f} | Perplexity : {:5.4f} | Time: {:5.0f} ms'.format(
                epoch, batch, data_size // opt.batch_size, lr, np.mean(total_loss), perplexity, elapsed_time))
            start = time.time()
            numerator = 0.0
            denominator = 0.0

    print('\nEpoch: {:5d} | Average loss: {:5.4f} | Average Perplexity : {:5.4f} | Average Accuracy : {:5.4f}'.format(
        epoch, np.mean(total_loss), perplexity, np.mean(accuracy)))

    return np.mean(total_loss), perplexity, np.mean(accuracy)


def validate_model(model, dataset, epoch, opt):
    print("----------------------------------Validation----------------------------------")
    model.eval()

    # Ignore the padding tokens for the loss computation
    pad_index = dataset.vocabulary.word2token['-PAD-']
    criterion_loss = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')

    vocab_size = len(dataset.vocabulary)
    data_size = len(dataset.val_data)
    start = time.time()
    total_loss = []
    numerator = 0.0
    denominator = 0.0
    accuracy = 0.0

    with torch.no_grad():
        for batch, idx in enumerate(range(0, data_size - 1, opt.test_batch)):
            source, target, sentence_len = dataset.load_data('val', idx, opt.test_batch)
            if source is None:
                continue
            if torch.cuda.is_available():
                source = source.cuda()
                target = target.cuda()

            embeddings, logv, mean, std = model.encode(source, opt.test_batch)
            output, _ = model.decode(embeddings, mean, std, opt.test_batch, num_samples=opt.sample_size)
            output = output.view(opt.test_batch * opt.seq_length, vocab_size)
            target = target.view(opt.test_batch * opt.seq_length)
            NLL_loss = criterion_loss(output, target)
            # Get the KL loss term and the weight
            KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
            KL_weight = kl_weight_function(anneal=opt.anneal, step=batch)

            # Compute the validation loss
            loss = (NLL_loss + KL_weight * KL_loss)
            total_loss.append(loss.cpu().item() / (opt.test_batch * opt.seq_length))

            # Compute the perplexity
            numerator += loss.item()
            denominator += np.sum(sentence_len)

            # Compute the word prediction accuracy
            output = output.view(opt.test_batch, -1, vocab_size)
            target = target.view(opt.test_batch, -1)
            accuracy += compute_accuracy(output, target, sentence_len, pad_index)

    loss = np.sum(total_loss) / data_size
    per_word_ppl = np.exp(numerator / denominator)
    accuracy = accuracy / batch
    elapsed_time = (time.time() - start) * 1000 / opt.print_interval
    print('Epoch: {:5d} | loss: {:5.4f} | Perplexity : {:5.4f} | Accuracy : {:5.4f} | Time: {:5.0f} ms'.format(
        epoch, loss, per_word_ppl, accuracy, elapsed_time))

    return loss, per_word_ppl, accuracy


def test_model(model, dataset, epoch, opt):
    print("----------------------------------Testing----------------------------------")
    model.eval()

    # Ignore the padding tokens for the loss computation
    pad_index = dataset.vocabulary.word2token['-PAD-']
    criterion_loss = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')

    vocab_size = len(dataset.vocabulary)
    data_size = len(dataset.test_data)
    start = time.time()
    total_loss = []
    numerator = 0.0
    denominator = 0.0
    accuracy = 0.0

    with torch.no_grad():
        for batch, idx in enumerate(range(0, data_size - 1, opt.test_batch)):
            source, target, sentence_len = dataset.load_data('test', idx, opt.test_batch)
            if source is None:
                continue
            if torch.cuda.is_available():
                source = source.cuda()
                target = target.cuda()

            embeddings, logv, mean, std = model.encode(source, opt.test_batch)
            output, _ = model.decode(embeddings, mean, std, opt.test_batch, num_samples=opt.sample_size)
            output = output.view(opt.test_batch * opt.seq_length, vocab_size)
            target = target.view(opt.test_batch * opt.seq_length)
            NLL_loss = criterion_loss(output, target)
            # Get the KL loss term and the weight
            KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
            KL_weight = kl_weight_function(anneal=opt.anneal, step=batch)

            # Compute the validation loss
            loss = (NLL_loss + KL_weight * KL_loss)
            total_loss.append(loss.cpu().item() / (opt.test_batch * opt.seq_length))

            # Compute the perplexity
            numerator += loss.item()
            denominator += np.sum(sentence_len)

            # Compute the word prediction accuracy
            output = output.view(opt.test_batch, -1, vocab_size)
            target = target.view(opt.test_batch, -1)
            accuracy += compute_accuracy(output, target, sentence_len, pad_index)

    loss = np.sum(total_loss) / data_size
    per_word_ppl = np.exp(numerator / denominator)
    accuracy = accuracy / batch
    elapsed_time = (time.time() - start) * 1000 / opt.print_interval
    print('Epoch: {:5d} | loss: {:5.4f} | Perplexity : {:5.4f} | Accuracy : {:5.4f} | Time: {:5.0f} ms'.format(
        epoch, loss, per_word_ppl, accuracy, elapsed_time))


def compute_accuracy(output, target, sentence_len, pad_index):
    output = torch.argmax(output, dim=2)
    correct = (target == output).float()
    # Ignore the padded indices
    correct[target == pad_index] = 0
    accuracy = torch.sum(correct) / np.sum(sentence_len)

    return accuracy.item()


def generate_sentences(model, dataset, sentence_len, method='multi'):
    print('\n\n----------------------------------Sentence Generation----------------------------------')
    model.eval()

    pad_idx = dataset.vocabulary.word2token['-PAD-']
    sos_idx = dataset.vocabulary.word2token['-SOS-']
    eos_idx = dataset.vocabulary.word2token['-EOS-']
    sentence = []

    tokens, z = model.inference(None, 1, sentence_len, pad_idx, sos_idx, eos_idx, method)

    for word_idx in tokens[0]:
        word = dataset.vocabulary.vocab[word_idx]
        sentence.append(word)

    final_sentence = '\t'
    for word in sentence:
        if word == '-SOS-':
            final_sentence = final_sentence + '\t'
        elif word == '-EOS-':
            final_sentence = final_sentence + ' .\n'
        elif word == '-PAD-':
            final_sentence = final_sentence
        else:
            final_sentence = final_sentence + ' ' + word
    print(final_sentence)

    return final_sentence

def generate_homotopy(model, dataset, opt, sentence_len, steps, method):
    print('\n\n----------------------------------Homotopy----------------------------------')
    model.eval()

    pad_idx = dataset.vocabulary.word2token['-PAD-']
    sos_idx = dataset.vocabulary.word2token['-SOS-']
    eos_idx = dataset.vocabulary.word2token['-EOS-']

    z1 = np.array(torch.randn([opt.latent_size]))
    z2 = np.array(torch.randn([opt.latent_size]))
    z = intepolation(start=z1, end=z2, steps=steps)
    z = torch.Tensor(z).cuda() if torch.cuda.is_available() else torch.Tensor(z)

    tokens, _ = model.inference(z, z.size(0), sentence_len, pad_idx, sos_idx, eos_idx, method)

    homotopy = [str()] * len(tokens)
    for i, sentence in enumerate(tokens):
        for word_idx in sentence:
            if word_idx == pad_idx:
                break
            elif word_idx == sos_idx:
                homotopy[i] += '\t'
            elif word_idx == eos_idx:
                homotopy[i] += '\n'
            else:
                homotopy[i] += dataset.vocabulary.vocab[word_idx] + " "

    for i in range(len(homotopy)):
        print(homotopy[i])


def intepolation(start, end, steps):
    interpol = np.zeros((start.shape[0], steps + 2))
    for dim, (s, e) in enumerate(zip(start, end)):
        interpol[dim] = np.linspace(s, e, steps + 2)

    return interpol.T
