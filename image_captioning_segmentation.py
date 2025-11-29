"""
Image Captioning Project — Full code in one file

Save this file as image_captioning_project.py and run with Python 3.8+ (recommend PyTorch + CUDA if available).

Expected dataset CSV: /mnt/data/sagid_notest.csv (as you uploaded). The CSV must contain at least two columns:
- image: path to image file (absolute path or relative to the CSV location)
- caption: a textual caption (if there are multiple captions per image, multiple rows are fine)

Files created by the script: model checkpoints (encoder.pth, decoder.pth), tokenizer saved as vocab.pkl

How to run (examples):
1) Preprocess & build dataset & vocab + start training (default):
   python image_captioning_project.py --csv /mnt/data/sagid_notest.csv --images_root /mnt/data/images --epochs 10 --batch_size 64

2) Run inference on a single image:
   python image_captioning_project.py --infer --image_path /path/to/image.jpg --checkpoint decoder.pth --vocab vocab.pkl

Notes:
- This is a complete, self-contained example using PyTorch, torchvision and NLTK (for tokenization).
- The script uses a pretrained ResNet-50 as encoder (from torchvision.models). If your environment has no internet, ensure torchvision has pretrained weights available or set pretrained=False (but training will be harder).

"""

# Imports
import os
import sys
import argparse
import random
import math
import time
import json
import pickle
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# --------------------------
# Utilities & Tokenizer
# --------------------------
class Vocabulary(object):
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v:k for k,v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        # Simple tokenizer; you can replace with nltk.word_tokenize if desired
        return text.lower().strip().split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            tokens = Vocabulary.tokenize(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = Vocabulary.tokenize(text)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokenized_text]

# --------------------------
# Dataset
# --------------------------
class CaptionDataset(Dataset):
    def __init__(self, csv_path, images_root, vocabulary=None, transform=None, freq_threshold=5, max_len=50):
        import csv
        self.csv_path = csv_path
        self.images_root = images_root
        self.transform = transform
        self.max_len = max_len

        self.items = []  # list of (image_path, caption)
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Try to detect columns
            cols = reader.fieldnames
            if not cols:
                raise ValueError('CSV has no header')
            # find columns named like image, img, filepath and caption, caption_text
            image_col = None
            caption_col = None
            for c in cols:
                lc = c.lower()
                if image_col is None and any(k in lc for k in ['image', 'img', 'file', 'filepath', 'path']):
                    image_col = c
                if caption_col is None and any(k in lc for k in ['caption', 'cap', 'text', 'description']):
                    caption_col = c
            if image_col is None or caption_col is None:
                raise ValueError(f'Could not auto-detect image and caption columns. Found columns: {cols}')

            for row in reader:
                img_path = row[image_col].strip()
                caption = row[caption_col].strip()
                if not os.path.isabs(img_path):
                    # relative path from images_root, or relative to CSV
                    candidate1 = os.path.join(images_root, img_path)
                    candidate2 = os.path.join(os.path.dirname(csv_path), img_path)
                    if os.path.exists(candidate1):
                        img_path = candidate1
                    elif os.path.exists(candidate2):
                        img_path = candidate2
                    # else leave as-is (user might provide absolute paths or other layout)

                self.items.append((img_path, caption))

        # build vocabulary if not supplied
        all_captions = [c for _, c in self.items]
        if vocabulary is None:
            self.vocab = Vocabulary(freq_threshold=freq_threshold)
            self.vocab.build_vocabulary(all_captions)
        else:
            self.vocab = vocabulary

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, caption = self.items[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f'Could not open image {img_path}: {e}')

        if self.transform is not None:
            image = self.transform(image)

        # numericalize caption and add <start> and <end>
        numericalized = [self.vocab.stoi['<start>']] + self.vocab.numericalize(caption) + [self.vocab.stoi['<end>']]
        if len(numericalized) > self.max_len:
            numericalized = numericalized[:self.max_len-1] + [self.vocab.stoi['<end>']]

        caption_tensor = torch.tensor(numericalized)
        return image, caption_tensor

# Collate fn for variable length captions
def collate_fn(batch):
    # batch: list of (image, caption_tensor)
    images = [item[0].unsqueeze(0) for item in batch]
    images = torch.cat(images, dim=0)
    captions = [item[1] for item in batch]
    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded[i, :end] = cap
    return images, padded, lengths

# --------------------------
# Model: Encoder (CNN) + Attention + Decoder (LSTM)
# --------------------------
class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14, embed_size=256, train_cnn=False):
        super(EncoderCNN, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = models.resnet50(pretrained=True)
        # remove linear and pool layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune(train_cnn)
        # project CNN output to embedding space
        self.conv = nn.Conv2d(2048, embed_size, 1)
        self.bn = nn.BatchNorm2d(embed_size)

    def forward(self, images):
        features = self.resnet(images)  # (batch, 2048, feat_h, feat_w)
        features = self.adaptive_pool(features)  # (batch, 2048, enc_h, enc_w)
        features = self.conv(features)  # (batch, embed_size, enc_h, enc_w)
        features = self.bn(features)
        # flatten to (batch, num_pixels, embed_size)
        batch_size, embed_size, enc_h, enc_w = features.size()
        features = features.permute(0, 2, 3, 1)
        features = features.view(batch_size, -1, embed_size)
        return features  # (batch, num_pixels, embed_size)

    def fine_tune(self, fine_tune=False):
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tune, enable last conv layers
        if fine_tune:
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = True

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (batch, num_pixels, encoder_dim)
        # decoder_hidden: (batch, decoder_dim)
        att1 = self.encoder_att(encoder_out)  # (batch, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # (batch, 1, attention_dim)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (batch, num_pixels)
        alpha = self.softmax(att)  # (batch, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch, encoder_dim)
        return attention_weighted_encoding, alpha

class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=256, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # gating scalar
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        # encoder_out: (batch, num_pixels, encoder_dim)
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        # encoder_out: (batch, num_pixels, encoder_dim)
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size

        # sort by caption_lengths descending (not strictly required here but common)
        caption_lengths_sorted, sort_ind = torch.sort(torch.tensor(caption_lengths), descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)  # (batch, max_len, embed_dim)

        h, c = self.init_hidden_state(encoder_out)  # (batch, decoder_dim)

        max_length = max(caption_lengths_sorted).item()
        predictions = torch.zeros(batch_size, max_length, vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_length, encoder_out.size(1)).to(encoder_out.device)

        for t in range(max_length):
            batch_size_t = sum([l > t for l in caption_lengths_sorted])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            emb_t = embeddings[:batch_size_t, t, :]
            input_lstm = torch.cat([emb_t, attention_weighted_encoding], dim=1)
            h_t, c_t = self.decode_step(input_lstm, (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h_t))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

            h[:batch_size_t] = h_t
            c[:batch_size_t] = c_t

        return predictions, encoded_captions, caption_lengths_sorted, alphas, sort_ind

# --------------------------
# Training & Helpers
# --------------------------

def train_one_epoch(encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, dataloader, device, epoch, print_every=100):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    start = time.time()
    for i, (imgs, caps, lengths) in enumerate(dataloader):
        imgs = imgs.to(device)
        caps = caps.to(device)
        lengths = lengths

        # forward
        features = encoder(imgs)  # (batch, num_pixels, encoder_dim)
        preds, caps_sorted, cap_lengths, alphas, sort_ind = decoder(features, caps, lengths)

        # Remove timesteps that we didn't predict for
        targets = caps_sorted
        # compute loss (pack targets)
        # preds: (batch, max_len, vocab_size)
        preds = preds.view(-1, preds.size(2))
        targets = targets[:, :preds.size(0) // targets.size(0)].contiguous().view(-1)
        # But above is error-prone. We'll compute loss per timestep respecting lengths instead.

        # compute mask-based loss
        loss = 0.0
        batch_size = targets.size(0)
        # simpler: iterate over batch and lengths (less efficient but clearer)
        loss = 0.0
        idx = 0
        for b in range(len(cap_lengths)):
            L = cap_lengths[b]
            for t in range(L):
                pred_t = preds[b*preds.size(0)//len(cap_lengths) + t] if False else None
        # We'll instead use packed approach: create mask and compute cross-entropy
        max_len = caps.size(1)
        preds = preds.view(caps.size(0), max_len, -1)
        # shift targets to remove <start>
        targets = caps[:, :max_len]
        # build mask
        mask = (targets != 0).float()
        preds_flat = preds.reshape(-1, preds.size(2))
        targets_flat = targets.reshape(-1)
        loss = criterion(preds_flat, targets_flat)

        # backward
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # clip
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.)
        if encoder_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        total_loss += loss.item()

        if (i+1) % print_every == 0:
            elapsed = time.time() - start
            print(f'Epoch [{epoch}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Time: {elapsed:.1f}s')
            start = time.time()

    return total_loss / len(dataloader)

# Simple evaluation: greedy decode
def generate_caption(encoder, decoder, image, vocab, device, max_len=50):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        features = encoder(image)  # (1, num_pixels, encoder_dim)
        h, c = decoder.init_hidden_state(features)
        sampled_ids = []
        inputs = torch.tensor([vocab.stoi['<start>']]).to(device)
        embeddings = decoder.embedding(inputs)
        for _ in range(max_len):
            attention_weighted_encoding, alpha = decoder.attention(features, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            input_lstm = torch.cat([embeddings, attention_weighted_encoding.unsqueeze(0)], dim=1) if False else torch.cat([decoder.embedding(inputs), attention_weighted_encoding.unsqueeze(0)], dim=1)
            h, c = decoder.decode_step(input_lstm.squeeze(1), (h, c))
            outputs = decoder.fc(h)
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())
            inputs = predicted
            embeddings = decoder.embedding(inputs)
            if predicted.item() == vocab.stoi['<end>']:
                break
    # convert ids to words
    words = []
    for id in sampled_ids:
        word = vocab.itos.get(id, '<unk>')
        if word == '<end>':
            break
        words.append(word)
    return ' '.join(words)

# --------------------------
# Main routine
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='/mnt/data/sagid_notest.csv', help='path to csv file')
    parser.add_argument('--images_root', type=str, default='/mnt/data/images', help='root folder where images are stored')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--attention_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--freq_threshold', type=int, default=5)
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--checkpoint_decoder', type=str, default=None)
    parser.add_argument('--checkpoint_encoder', type=str, default=None)
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    if args.infer:
        if args.image_path is None or args.checkpoint_decoder is None or not os.path.exists(args.vocab_path):
            print('For inference, provide --image_path, --checkpoint_decoder and ensure vocab exists at --vocab_path')
            return
        # load vocab
        with open(args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        encoder = EncoderCNN(embed_size=args.embed_size).to(device)
        decoder = DecoderWithAttention(attention_dim=args.attention_dim, embed_dim=args.embed_size, decoder_dim=args.decoder_dim, vocab_size=len(vocab), encoder_dim=args.embed_size).to(device)
        if args.checkpoint_encoder and os.path.exists(args.checkpoint_encoder):
            encoder.load_state_dict(torch.load(args.checkpoint_encoder, map_location=device))
        decoder.load_state_dict(torch.load(args.checkpoint_decoder, map_location=device))
        # load image
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])
        image = Image.open(args.image_path).convert('RGB')
        image = transform(image)
        caption = generate_caption(encoder, decoder, image, vocab, device)
        print('Generated caption:', caption)
        return

    # Training mode
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f'CSV file not found at {args.csv}')

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])

    dataset = CaptionDataset(args.csv, args.images_root, transform=transform, freq_threshold=args.freq_threshold)
    vocab = dataset.vocab
    print('Vocab size:', len(vocab))

    # save vocab
    with open(args.vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print('Saved vocab to', args.vocab_path)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    encoder = EncoderCNN(embed_size=args.embed_size).to(device)
    decoder = DecoderWithAttention(attention_dim=args.attention_dim, embed_dim=args.embed_size, decoder_dim=args.decoder_dim, vocab_size=len(vocab), encoder_dim=args.embed_size).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    params = list(decoder.parameters()) + list(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.lr/10)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for epoch in range(1, args.epochs+1):
        loss = train_one_epoch(encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, dataloader, device, epoch)
        print(f'Epoch {epoch} finished. Avg loss: {loss:.4f}')
        # save checkpoints
        torch.save(decoder.state_dict(), os.path.join(args.save_dir, f'decoder_epoch{epoch}.pth'))
        torch.save(encoder.state_dict(), os.path.join(args.save_dir, f'encoder_epoch{epoch}.pth'))

    print('Training finished. Models saved in', args.save_dir)

if __name__ == '__main__':
    main()
