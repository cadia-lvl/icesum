#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This code is largely based on nnsum: https://github.com/kedz/nnsum
import torch
from torch.utils.data import Dataset, DataLoader
import tokenizer


class SumDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        super(SumDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            # batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            collate_fn=self._collate_fn)

    class SumBatch(object):
        def __init__(self, id, document, num_sentences, sentence_lengths, sentence_texts, pretty_sentence_lengths):
            self.id = id
            self.document = document
            self.num_sentences = num_sentences
            self.sentence_lengths = sentence_lengths
            self.sentence_texts = sentence_texts
            self.pretty_sentence_lengths = pretty_sentence_lengths

        def to(self, device=-1):
            if device < 0:
                return self
            else:
                document = self.document.to(device)

                num_sentences = self.num_sentences.to(device)
                sentence_lengths = self.sentence_lengths.to(device)
                return self.__class__(self.id, document, num_sentences, sentence_lengths, self.sentence_texts,
                                      self.pretty_sentence_lengths)

    def _collate_fn(self, batch):
        pad = self.dataset.vocab.pad_index

        batch.sort(key=lambda x: x["num_sentences"], reverse=True)
        ids = [item["id"] for item in batch]
        documents = batch_pad_and_stack_matrix([item["document"] for item in batch], pad)

        num_sentences = torch.LongTensor([item["num_sentences"] for item in batch])
        sentence_lengths = batch_pad_and_stack_vector([item["sentence_lengths"] for item in batch], 0)

        sentence_texts = [item["pretty_sentences"] for item in batch]
        pretty_sentence_lengths = [item["pretty_sentence_lengths"] for item in batch]

        return self.SumBatch(ids, documents, num_sentences, sentence_lengths, sentence_texts, pretty_sentence_lengths)


class SumDataset(Dataset):
    def __init__(self, vocab, document, sentence_limit=None):
        self._vocab = vocab
        self._sentence_limit = sentence_limit
        self._inputs = [document]

    @property
    def vocab(self):
        return self._vocab

    @property
    def sentence_limit(self):
        return self._sentence_limit

    def __len__(self):
        return len(self._inputs)

    def _read_inputs(self, data):
        # Get the length of the document in sentences.
        doc_size = len(data["inputs"])

        # Get the token lengths of each sentence and the maximum sentence
        # sentence length. If sentence_limit is set, truncate document
        # to that length.
        if self.sentence_limit:
            doc_size = min(self.sentence_limit, doc_size)

        sent_sizes = torch.LongTensor([len(sent["tokens"]) for sent in data["inputs"][:doc_size]])
        sent_size = sent_sizes.max().item()

        # Create a document matrix of size doc_size x sent_size. Fill in
        # the word indices with vocab.
        document = torch.LongTensor(doc_size, sent_size).fill_(self.vocab.pad_index)
        for s, sent in enumerate(data["inputs"][:doc_size]):
            for t, token in enumerate(sent["tokens"]):
                document[s, t] = self.vocab[token.lower()]

        # Get pretty sentences that are detokenized and their lengths for
        # generating the actual sentences.
        pretty_sentences = [sent["text"] for sent in data["inputs"][:doc_size]]
        pretty_sentence_lengths = torch.LongTensor([len(sent.split()) for sent in pretty_sentences])

        output = {
            "id": data["id"],
            "num_sentences": doc_size,
            "sentence_lengths": sent_sizes,
            "document": document,
            "pretty_sentences": pretty_sentences,
            "pretty_sentence_lengths": pretty_sentence_lengths
        }

        return output

    def __getitem__(self, index):
        raw_inputs_data = self._inputs[index]
        inp_data = self._read_inputs(raw_inputs_data)

        data = zip(inp_data["document"],
                   inp_data["sentence_lengths"],
                   inp_data["pretty_sentences"],
                   inp_data["pretty_sentence_lengths"])

        for isent, slen, psent, pslen in data:
            try:
                assert isent.tolist().index(0) == slen
            except ValueError:
                assert isent.size(0) == slen

        return inp_data


def batch_pad_and_stack_matrix(tensors, pad):
    assert len(set([t.dim() for t in tensors])) == 1

    sizes = torch.stack([torch.LongTensor([*t.size()]) for t in tensors])
    max_sizes, _ = sizes.max(0)

    batch_size = len(tensors)
    batch_tensor = tensors[0].new(batch_size, *max_sizes).fill_(pad)

    for t, tsr in enumerate(tensors):
        tslice = batch_tensor[t,:tsr.size(0),:tsr.size(1)]
        tslice.copy_(tsr)
    return batch_tensor


def batch_pad_and_stack_vector(tensors, pad):
    max_size = max([t.size(0) for t in tensors])

    batch_size = len(tensors)
    batch_tensor = tensors[0].new(batch_size, max_size).fill_(pad)

    for t, tsr in enumerate(tensors):
        tslice = batch_tensor[t, :tsr.size(0)]
        tslice.copy_(tsr)
    return batch_tensor


class Summarizer(object):
    def __init__(self, model_path, gpu=-1):
        self.model = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.gpu = gpu

        if gpu > -1:
            self.model.cuda(gpu)

        self.vocab = self.model.embeddings.vocab
        self.model.eval()

        self.labels = []

    def predict(self, text, summary_length, sentence_limit=None):
        # if loader_workers is None:
        #     loader_workers = min(16, cpu_count())

        text = text.strip()
        sentences = [s for s in tokenizer.split_into_sentences(text)]
        tokenized = [s.split() for s in sentences]
        detokenized = detokenize(text.split(), tokenized)

        doc_inputs = [
            {
                "text": sentence,
                "sentence_id": num,
                "pos": [],
                "word_count": len(tokens),
                "tokens": tokens
            }

            for num, (sentence, tokens) in enumerate(zip(detokenized, tokenized), 1)
        ]

        doc = {"id": "doc", "inputs": doc_inputs}
        data = SumDataset(self.vocab, doc, sentence_limit=sentence_limit)
        loader = SumDataLoader(data, batch_size=1, num_workers=0)

        with torch.no_grad():
            for step, batch in enumerate(loader, 1):
                batch = batch.to(self.gpu)
                texts = self.model.predict(batch, max_length=summary_length)

                # Make sure that the sentences are in the correct order
                summary_sentences = set(texts[0])
                summary = " ".join([s for s in detokenized if s in summary_sentences])

                # labels = [int(s in set(texts[0])) for s in sentences]
                return summary


def detokenize(words, sentences):
    tokens = (t for s in sentences for t in s)
    spaces = []

    for word in words:
        token = next(tokens)

        while token != word:
            spaces.append(0)
            token += next(tokens)

        spaces.append(1)

    sp = iter(spaces)

    detokenized = []
    for sentence in sentences:
        sent = [t + " " if space else t for t, space in zip(sentence, sp)]
        detokenized.append("".join(sent).strip())

    return detokenized


def main():
    pass


if __name__ == '__main__':
    main()
