# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import modeling
import optimization
import tokenization
import six
import tensorflow as tf
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")


flags.DEFINE_string(
    "eval_examples_file", "eval_examples.json",
    "eval_examples_file")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer(
    "num_of_eval_samples", 5000,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 2.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("max_steps_without_decrease", 2500,
                     "maximum number of training steps with no decrease in the given metric.")

flags.DEFINE_integer("save_checkpoints_steps", 500,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")

VERY_NEGATIVE_NUMBER = -1e29

class CailExample(object):
    """A single training/test example for simple sequence classification.

       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False,
                 y_or_n=2):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.y_or_n = y_or_n

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % self.start_position
        if self.start_position:
            s += ", end_position: %d" % self.end_position
        if self.start_position:
            s += ", is_impossible: %r" % self.is_impossible
        if self.orig_answer_text:
            s += ", answer: %r" % (tokenization.printable_text(self.orig_answer_text))
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 unk_mask,
                 yes_mask,
                 no_mask,
                 extractive_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 y_or_n=2):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.unk_mask = unk_mask
        self.yes_mask = yes_mask
        self.no_mask = no_mask
        self.extractive_mask = extractive_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.y_or_n = y_or_n


def customize_tokenizer(text, do_lower_case=False):
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
    temp_x = ""
    text = tokenization.convert_to_unicode(text)
    for c in text:
        if tokenizer._is_chinese_char(ord(c)) or tokenization._is_punctuation(c) \
                or tokenization._is_whitespace(c) or tokenization._is_control(c):
            temp_x += " " + c + " "
        else:
            temp_x += c
    if do_lower_case:
        temp_x = temp_x.lower()
    return temp_x.split()


class ChineseFullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=False):
        self.vocab = tokenization.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab=self.vocab)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        split_tokens = []
        for token in customize_tokenizer(text, do_lower_case=self.do_lower_case):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return tokenization.convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return tokenization.convert_by_vocab(self.inv_vocab, ids)


def read_cail_examples(input_file, is_training):
    """Read a cail json file into a list of CailExample."""
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            raw_doc_tokens = customize_tokenizer(paragraph_text, do_lower_case=FLAGS.do_lower_case)
            doc_tokens = []
            char_to_word_offset = []
            k = 0
            temp_word = ""
            for c in paragraph_text:
                if tokenization._is_whitespace(c):
                    char_to_word_offset.append(k - 1)
                    continue
                else:
                    temp_word += c
                    char_to_word_offset.append(k)
                if FLAGS.do_lower_case:
                    temp_word = temp_word.lower()
                if temp_word == raw_doc_tokens[k]:
                    doc_tokens.append(temp_word)
                    temp_word = ""
                    k += 1
            assert k == len(raw_doc_tokens)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                yes_or_no = 2
                if is_training:
                    is_impossible = qa["is_impossible"]
                    if is_impossible == "true":
                        is_impossible = True
                    elif is_impossible == "false":
                        is_impossible = False
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        if answer_offset >= 0:
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = "".join(
                                doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = "".join(
                                tokenization.whitespace_tokenize(orig_answer_text))
                            if FLAGS.do_lower_case:
                                cleaned_answer_text = cleaned_answer_text.lower()
                            if actual_text.find(cleaned_answer_text) == -1:
                                tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                                   actual_text, cleaned_answer_text)
                                # tf.logging.warning("raw tokens: %s",
                                #                    raw_doc_tokens)
                                # tf.logging.warning("tokens: %s",
                                #                    doc_tokens)
                                # tf.logging.warning("char_to_word_offset: %s",
                                #                    char_to_word_offset)
                                # tf.logging.warning("start_index: %s",
                                #                    str(answer_offset))
                                # tf.logging.warning("end_index: %s",
                                #                    str(answer_offset))
                                # tf.logging.warning("start_position: %s",
                                #                    str(start_position))
                                # tf.logging.warning("end_position: %s",
                                #                    str(end_position))
                                continue
                        else:
                            start_position = -1
                            end_position = -1
                            if orig_answer_text == "YES":
                                yes_or_no = 0
                            elif orig_answer_text == "NO":
                                yes_or_no = 1
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = CailExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                    y_or_n=yes_or_no)
                examples.append(example)

    return examples


def save_examples(save_file, examples):
    save_example = []
    with tf.gfile.Open(save_file, "w") as writer:
        for example in examples:
            qas_id = example.qas_id
            question_text = example.question_text
            doc_tokens = example.doc_tokens
            orig_answer_text = example.orig_answer_text
            start_position = example.start_position
            end_position = example.end_position
            is_impossible = example.is_impossible
            y_or_n = example.y_or_n

            save_example.append({"qas_id": qas_id,
                                 "question_text": question_text,
                                 "doc_tokens": doc_tokens,
                                 "orig_answer_text": orig_answer_text,
                                 "start_position": start_position,
                                 "end_position": end_position,
                                 "is_impossible": is_impossible,
                                 "y_or_n": y_or_n})
        writer.write(json.dumps(save_example, indent=4, ensure_ascii=False) + "\n")


def load_examples(input_file):
    examples = []
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)
        for each_sample in input_data:
            examples.append(CailExample(
                    qas_id=each_sample["qas_id"],
                    question_text=each_sample["question_text"],
                    doc_tokens=each_sample["doc_tokens"],
                    orig_answer_text=each_sample["orig_answer_text"],
                    start_position=each_sample["start_position"],
                    end_position=each_sample["end_position"],
                    is_impossible=each_sample["is_impossible"],
                    y_or_n=each_sample["y_or_n"]))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    ch_tokenizer = ChineseFullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        # A B C
        # [0]=0 len=1
        # [1]=1 len=2
        # [2]=2 len=3
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = ch_tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible and example.y_or_n == 2:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            unk_mask = 0
            yes_mask = 0
            no_mask = 0
            extractive_mask = 1
            if is_training and not example.is_impossible and example.y_or_n == 2:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0
                unk_mask = 1
                extractive_mask = 0

            if is_training and example.y_or_n != 2:
                start_position = 0
                end_position = 0
                extractive_mask = 0
                if example.y_or_n == 0:
                    yes_mask = 1
                elif example.y_or_n == 1:
                    no_mask = 1

            if example_index < 20:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                tf.logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    tf.logging.info("impossible example")
                if is_training and example.y_or_n != 2:
                    if example.y_or_n == 0:
                        answer_text = "YES"
                    else:
                        answer_text = "NO"
                    tf.logging.info("answer: %s" % (answer_text))
                if is_training and not example.is_impossible and example.y_or_n == 2:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    tf.logging.info("start_position: %d" % (start_position))
                    tf.logging.info("end_position: %d" % (end_position))
                    tf.logging.info(
                        "answer: %s" % (tokenization.printable_text(answer_text)))

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                unk_mask=unk_mask,
                yes_mask=yes_mask,
                no_mask=no_mask,
                extractive_mask=extractive_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible,
                y_or_n=example.y_or_n)

            # Run callback
            output_fn(feature)

            unique_id += 1


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def multi_linear_layer(x, layers, hidden_size, output_size, activation=None):

    if layers <= 0:
        return x
    for i in range(layers - 1):
        with tf.variable_scope("linear_layer" + str(i + 1)):
            w = tf.get_variable(
                "w", [hidden_size, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            b = tf.get_variable(
                "b", [hidden_size], initializer=tf.zeros_initializer())
            x = tf.nn.bias_add(tf.matmul(x, w), b)
            if activation == "relu":
                x = tf.nn.relu(x)
            elif activation == "tanh":
                x = tf.tanh(x)
            elif activation == "gelu":
                x = modeling.gelu(x)
    with tf.variable_scope("linear_layer" + str(layers)):
        w = tf.get_variable(
            "w", [hidden_size, output_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        b = tf.get_variable(
            "b", [output_size], initializer=tf.zeros_initializer())
        x = tf.nn.bias_add(tf.matmul(x, w), b)
    return x


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]
    final_hidden_matrix = tf.reshape(final_hidden,
                                     [batch_size * seq_length, hidden_size])
    pooled_output = model.get_pooled_output()

    n_layers = 2
    activation = 'relu'
    with tf.variable_scope("answer_logits"):
        logits = multi_linear_layer(final_hidden_matrix, n_layers, hidden_size, 2, activation=activation)
    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)
    # [b,s]
    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    # print(unk_yes_no_logits)
    with tf.variable_scope("unk_yes_no"):
        # [b,3]
        unk_yes_no_logits = multi_linear_layer(pooled_output, n_layers, hidden_size, 3, activation=activation)
    # [3,b]
    unstacked_logits1 = tf.unstack(unk_yes_no_logits, axis=1)
    # [b]
    unk_logits, yes_logits, no_logits = unstacked_logits1
    # 三个logits的shape都是[b,1]
    unk_logits = tf.expand_dims(unk_logits, 1)
    yes_logits = tf.expand_dims(yes_logits, 1)
    no_logits = tf.expand_dims(no_logits, 1)
    # 前两个shape是[b,s],后三个是[b,1]
    return (start_logits, end_logits, unk_logits, yes_logits, no_logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (start_logits, end_logits, unk_logits, yes_logits, no_logits) = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            tf.logging.info("初始化参数")

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "unique_ids": unique_ids,
                "start_logits": start_logits,
                "end_logits": end_logits,
                "unk_logits": unk_logits,
                "yes_logits": yes_logits,
                "no_logits": no_logits
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)
        else:
            seq_length = modeling.get_shape_list(input_ids)[1]

            start_positions = features["start_positions"]
            end_positions = features["end_positions"]
            # 如果问题不可回答，unk_mask=1，否则为0
            unk_masks = features["unk_masks"]
            # 如果answer="YES"，yes_mask=1
            yes_masks = features["yes_masks"]
            # 如果answer="YES"，no_mask=1
            no_masks = features["no_masks"]
            # 如果该样本答案是"YES","NO",""三者之一，则extractive=0,否则为1
            extractive_masks = features["extractive_masks"]
            input_mask0 = tf.cast(input_mask, tf.float32)
            # [b,s]
            masked_start_logits = start_logits * input_mask0 + (1 - input_mask0) * VERY_NEGATIVE_NUMBER
            masked_end_logits = end_logits * input_mask0 + (1 - input_mask0) * VERY_NEGATIVE_NUMBER
            # [b]-->[b,s]
            start_masks = tf.one_hot(start_positions, depth=seq_length, dtype=tf.float32)
            end_masks = tf.one_hot(end_positions, depth=seq_length, dtype=tf.float32)
            # [b,s] = [b,s] * [b,1]
            start_masks = start_masks * tf.expand_dims(tf.cast(extractive_masks, tf.float32), axis=-1)
            end_masks = end_masks * tf.expand_dims(tf.cast(extractive_masks, tf.float32), axis=-1)

            # [b]-->[b,1]
            unk_masks = tf.expand_dims(tf.cast(unk_masks, tf.float32), axis=-1)
            yes_masks = tf.expand_dims(tf.cast(yes_masks, tf.float32), axis=-1)
            no_masks = tf.expand_dims(tf.cast(no_masks, tf.float32), axis=-1)
            # [b,s+3]
            new_start_masks = tf.concat([start_masks, unk_masks, yes_masks, no_masks], axis=1)
            new_end_masks = tf.concat([end_masks, unk_masks, yes_masks, no_masks], axis=1)
            # [b,s+3]
            new_start_logits = tf.concat([masked_start_logits, unk_logits, yes_logits, no_logits], axis=1)
            new_end_logits = tf.concat([masked_end_logits, unk_logits, yes_logits, no_logits], axis=1)

            # log（sum（exp（张量的各维数的元素）））：先对每个元素求exp，再求和，再求log
            # [b]
            # norm = log(exp(logit1)+exp(logit2)+...+exp(logit(s+3)))
            start_log_norm = tf.reduce_logsumexp(new_start_logits, axis=1)
            # 非正确标签所在位置的logit加上一个负无穷的数，再求exp
            # 约等于sum([0 0 ... exp(logit) ... 0 0 0 ])=exp(logit)
            # score约等于log(exp(logit))=logit,表示利用正确标签位置的ligit计算得分
            # [b]
            start_log_score = tf.reduce_logsumexp(
                new_start_logits + VERY_NEGATIVE_NUMBER * (1 - tf.cast(new_start_masks, tf.float32)), axis=1)
            # loss=norm-score
            # log_score - log_norm = log(exp(logit)) - log(exp(logit1)+exp(logit2)+...+exp(logit(s+3)))
            # = log(exp(logit)/(exp(logit1)+exp(logit2)+...+exp(logit(s+3))))
            # 即交叉熵
            start_loss = tf.reduce_mean(-(start_log_score - start_log_norm))

            end_log_norm = tf.reduce_logsumexp(new_end_logits, axis=1)
            end_log_score = tf.reduce_logsumexp(
                new_end_logits + VERY_NEGATIVE_NUMBER * (1 - tf.cast(new_end_masks, tf.float32)), axis=1)
            end_loss = tf.reduce_mean(-(end_log_score - end_log_norm))
            total_loss = (start_loss + end_loss) / 2.0

            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op)

            elif mode == tf.estimator.ModeKeys. EVAL:
                def my_metric_fn(s_loss, e_loss):
                    my_loss = [s_loss, e_loss]
                    mean, op = tf.metrics.mean(my_loss)
                    return mean, op
                eval_metrics = {
                    'eval_loss': my_metric_fn(start_loss, end_loss)
                }
                output_spec = tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=eval_metrics)

            else:
                raise ValueError(
                    "Only TRAIN, EVAL and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    if is_training:
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["unk_masks"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["yes_masks"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["no_masks"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["extractive_masks"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            batch_size = params["train_batch_size"]
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        else:
            batch_size = params["predict_batch_size"]
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits",
                                    "unk_logits", "yes_logits", "no_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file):
    """Write final predictions to the json file and log-odds of null if needed."""
    tf.logging.info("Writing predictions to: %s" % (output_prediction_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()

    pred_dict = []
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        max_unk_score = -100000000
        max_yes_score = -100000000
        max_no_score = -100000000
        # 通过doc_span方法一个原样本可能被切割成多个样本，对应不同的feature
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            unk_score = result.unk_logits[0] * 2
            if unk_score > max_unk_score:
                max_unk_score = unk_score
            yes_score = result.yes_logits[0] * 2
            if yes_score > max_yes_score:
                max_yes_score = yes_score
            no_score = result.no_logits[0] * 2
            if no_score > max_no_score:
                max_no_score = no_score
            # 根据logits值比较大的start index和end index构造候选的预测答案
            max_start_index = -1
            max_end_index = -1
            max_logits = -100000000
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    sum_logits = result.start_logits[start_index] + result.end_logits[end_index]
                    if sum_logits > max_logits:
                        max_logits = sum_logits
                        max_start_index = start_index
                        max_end_index = end_index
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=feature_index,
                    start_index=max_start_index,
                    end_index=max_end_index,
                    start_logit=result.start_logits[max_start_index],
                    end_logit=result.end_logits[max_end_index]))

        # 候选的预测答案按照start position和end position的logit之和从大到小排序
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        if len(prelim_predictions) > 0:
            pred = prelim_predictions[0]
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                final_text = final_text.replace(' ', '')    # 中文，要去除每个词之间的空格
            else:
                final_text = ""

            best = _NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit)

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        else:
            best = _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0)

        # predict "" iff the null score - the score of best non-null > threshold
        example_score = [best.start_logit+best.end_logit, max_unk_score, max_yes_score, max_no_score]
        max_index = np.argmax(example_score)
        if max_index == 0:
            all_predictions[example.qas_id] = best.text
        elif max_index == 1:
            all_predictions[example.qas_id] = ""
        elif max_index == 2:
            all_predictions[example.qas_id] = "YES"
        else:
            all_predictions[example.qas_id] = "NO"

    for my_id in all_predictions.keys():
        answer = all_predictions[my_id]
        pred_dict.append({"answer": answer, "id": my_id})
    tf.logging.info("Starting write result...")
    with tf.gfile.GFile(output_prediction_file, "w") as writer:
        writer.write('[')
        for obj in pred_dict[:-1]:
            json.dump(obj, writer, ensure_ascii=False, indent=4)
            writer.write(',')

        json.dump(pred_dict[-1], writer, ensure_ascii=False, indent=4)
        writer.write(']')
    tf.logging.info("Writing finished")


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if FLAGS.verbose_logging:
            tf.logging.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if FLAGS.verbose_logging:
            tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        if self.is_training:
            features["start_positions"] = create_int_feature([feature.start_position])
            features["end_positions"] = create_int_feature([feature.end_position])
            features["end_positions"] = create_int_feature([feature.end_position])
            features["unk_masks"] = create_int_feature([feature.unk_mask])
            features["yes_masks"] = create_int_feature([feature.yes_mask])
            features["no_masks"] = create_int_feature([feature.no_mask])
            features["extractive_masks"] = create_int_feature([feature.extractive_mask])

            impossible = 0
            if feature.is_impossible:
                impossible = 1
            features["is_impossible"] = create_int_feature([impossible])
            features["y_or_n"] = create_int_feature([feature.y_or_n])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def validate_flags_or_throw(bert_config):
    """Validate the input FLAGS or throw an exception."""
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if FLAGS.do_train:
        if not FLAGS.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")

    if FLAGS.do_predict:
        if not FLAGS.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
        raise ValueError(
            "The max_seq_length (%d) must be greater than max_query_length "
            "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
    FLAGS.predict_file = "../data/data.json"
    FLAGS.do_predict = True
    FLAGS.vocab_file = "chinese_L-12_H-768_A-12/vocab.txt"
    FLAGS.bert_config_file = "chinese_L-12_H-768_A-12/bert_config.json"
    FLAGS.max_seq_length = 512
    FLAGS.output_dir = "../result/"
    model_dir = "models/"
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    validate_flags_or_throw(bert_config)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    num_train_steps = None
    num_warmup_steps = None

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    hyparams = {"train_batch_size": FLAGS.train_batch_size, "predict_batch_size": FLAGS.predict_batch_size}
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=FLAGS.save_checkpoints_steps, keep_checkpoint_max=5)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=run_config,
        params=hyparams)

    eval_examples = read_cail_examples(
        input_file=FLAGS.predict_file, is_training=False)

    eval_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "predict.tf_record"),
        is_training=False)
    eval_features = []

    def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)

    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature)
    eval_writer.close()

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_input_fn = input_fn_builder(
        input_file=eval_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of
    # steps.
    all_results = []
    for result in estimator.predict(
            predict_input_fn, yield_single_examples=True):
        if len(all_results) % 1000 == 0:
            tf.logging.info("Processing example: %d" % (len(all_results)))
        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        unk_logits = [float(x) for x in result["unk_logits"].flat]
        yes_logits = [float(x) for x in result["yes_logits"].flat]
        no_logits = [float(x) for x in result["no_logits"].flat]
        all_results.append(
            RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits,
                unk_logits=unk_logits,
                yes_logits=yes_logits,
                no_logits=no_logits))

    output_prediction_file = os.path.join(FLAGS.output_dir, "result.json")

    write_predictions(eval_examples, eval_features, all_results,
                      FLAGS.n_best_size, FLAGS.max_answer_length,
                      FLAGS.do_lower_case, output_prediction_file)


if __name__ == "__main__":
    tf.app.run()
