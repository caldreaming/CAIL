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
"""Run BERT on SQuAD."""

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
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

flags = tf.flags

FLAGS = flags.FLAGS

# global_acc_list = []
## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "train_val_file", "train_val.json",
    "val file for training set. ")

flags.DEFINE_string(
    "dev_val_file", "dev_val.json",
    "val file for dev set. ")

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

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 128,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
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


class OqmrcExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 q_id,
                 question,
                 paragraph,
                 alters,
                 answer=None,
                 label=None):
        self.qas_id = q_id
        self.question = question
        self.paragraph = paragraph
        self.alters = alters
        self.answer = answer
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question))
        s += ", doc_tokens: %s" % (tokenization.printable_text(self.paragraph))
        s += ", alternatives: [%s]" % (" ".join(self.alters))
        if self.answer:
            s += ", answer: %s" % self.answer
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label


def read_oqmrc_examples(input_file, is_test):
    """Read a oqmrc json file into a list of OqmrcExample."""

    examples = []
    drop1 = 0
    drop2 = 0
    drop3 = 0
    drop4 = 0
    with tf.gfile.Open(input_file, "r") as reader:
        for line in reader:
            sample = json.loads(line)
            answer = None
            if not is_test:
                answer = sample["answer"]
                answer = answer.strip()
            if answer == "":
                drop1 += 1
                continue
            alternatives = sample['alternatives'].split("|")
            for i, alter in enumerate(alternatives):
                alternatives[i] = alter.strip()
            while "" in alternatives:
                alternatives.remove("")  # 删除空白的无效候选项
            if len(alternatives) != 3:
                drop2 += 1
                continue
            random.shuffle(alternatives)
            label = None
            ansnum = 0
            for i, alter in enumerate(alternatives):
                if alternatives[i] == answer:
                    label = i
                    ansnum += 1
            if ansnum > 1:
                drop3 += 1
                continue
            if not is_test and label is None:
                drop4 += 1
                continue
            paragraph_text = sample["passage"]
            qas_id = sample["query_id"]
            question_text = sample["query"]
            example = OqmrcExample(
                q_id=qas_id,
                question=question_text,
                paragraph=paragraph_text,
                alters=alternatives,
                answer=answer,
                label=label)
            examples.append(example)
    tf.logging.info("%d个样本答案为空白，%d个样本候选项不足，%d个样本多个候选项和答案相同，%d个样本找不到答案" %
                    (drop1, drop2, drop3, drop4))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 max_query_length, is_training,
                                 output_fn, val_file):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 10000000
    val_dict = {}
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        paragraph_tokens = tokenizer.tokenize(example.paragraph)
        alters_tokens = tokenizer.tokenize(",".join(example.alters))

        # The -4 accounts for [CLS], [SEP] and [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - len(alters_tokens) - 4

        if len(paragraph_tokens) > max_tokens_for_doc:
            paragraph_tokens = paragraph_tokens[0:max_tokens_for_doc]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)

        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in paragraph_tokens:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        for token in alters_tokens:
            tokens.append(token)
            segment_ids.append(2)
        tokens.append("[SEP]")
        segment_ids.append(2)

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

        if example_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % (unique_id))
            tf.logging.info("example_index: %s" % (example_index))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info(
                "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info(
                "label: %s" % example.label)
            if is_training:
                tf.logging.info(
                    "answer: %s" % (tokenization.printable_text(example.answer)))

        val_dict[unique_id] = example.label

        feature = InputFeatures(
            unique_id=unique_id,
            example_index=example_index,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label=example.label)

        # Run callback
        output_fn(feature)

        unique_id += 1
    with tf.gfile.Open(val_file, "w") as f:
        f.write(json.dumps(val_dict))


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

    final_hidden = model.get_pooled_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=2)
    batch_size = final_hidden_shape[0]
    hidden_size = final_hidden_shape[1]

    # 输出层：三分类
    output_weights = tf.get_variable(
        "cls/oqmrc/output_weights", [3, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "cls/oqmrc/output_bias", [3], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden,
                                     [batch_size, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, 3])
    # logits = tf.transpose(logits, [2, 0, 1])
    return logits
    # unstacked_logits = tf.unstack(logits, axis=0)

    # (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    # return (start_logits, end_logits)


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

        logits = create_model(
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
                "prediction": tf.argmax(logits, axis=-1),
            }
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        else:
            def compute_loss(logits, positions):
                one_hot_positions = tf.one_hot(
                    positions, depth=3, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                return loss

            label = features["label"]

            loss = compute_loss(logits, label)
            predicted_classes = tf.argmax(logits, axis=-1)
            accuracy = tf.metrics.accuracy(labels=label, predictions=predicted_classes, name='acc_op')

            # global global_acc_list
            # global_acc_list.append(accuracy)
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = optimization.create_optimizer(
                    loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

                metrics = {'accuracy': accuracy}
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    eval_metric_ops=metrics,
                    train_op=train_op)
            elif mode == tf.estimator.ModeKeys.EVAL:
                metrics = {'accuracy': accuracy}
                output_spec = tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            else:
                raise ValueError(
                    "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(input_file, seq_length, is_test, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    if not is_test:
        name_to_features["label"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
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

    def __init__(self, filename, is_test):
        self.filename = filename
        self.is_test = is_test
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

        if not self.is_test:
            features["label"] = create_int_feature([feature.label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def validate_flags_or_throw(bert_config):
    """Validate the input FLAGS or throw an exception."""
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
    # 设置日志等级
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    # 检查输入参数是否合法
    validate_flags_or_throw(bert_config)
    # 创建输出目录
    tf.gfile.MakeDirs(FLAGS.output_dir)
    # 创建分词器
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        # 每个样本包含[qid，问题（string），文章（string），候选项（list），答案（string），label(int)]
        train_examples = read_oqmrc_examples(
            input_file=FLAGS.train_file, is_test=False)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        # Pre-shuffle the input to avoid having to make a very large shuffle
        # buffer in in the `input_fn`.
        rng = random.Random(12345)
        rng.shuffle(train_examples)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    hyparams = {"train_batch_size": FLAGS.train_batch_size, "predict_batch_size": FLAGS.predict_batch_size}

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=os.path.join(FLAGS.output_dir, "models/"),
        params=hyparams)

    if FLAGS.do_train:
        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.
        train_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
            is_test=False)
        train_val_file = os.path.join(FLAGS.output_dir, FLAGS.train_val_file)
        convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            max_query_length=FLAGS.max_query_length,
            is_training=True,
            output_fn=train_writer.process_feature,
            val_file=train_val_file)
        train_writer.close()

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num orig examples = %d", len(train_examples))
        tf.logging.info("  Num split examples = %d", train_writer.num_features)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        del train_examples

        train_input_fn = input_fn_builder(
            input_file=train_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_test=False,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_predict:
        eval_examples = read_oqmrc_examples(
            input_file=FLAGS.predict_file, is_test=False)

        eval_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
            is_test=False)
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        dev_val_file = os.path.join(FLAGS.output_dir, FLAGS.dev_val_file)
        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            max_query_length=FLAGS.max_query_length,
            is_training=False,
            output_fn=append_feature,
            val_file=dev_val_file)
        eval_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_test=False,
            is_training=False,
            drop_remainder=False)
        # global global_acc_list
        # global_acc_list = []
        all_results = []
        for result in estimator.predict(
                predict_input_fn, yield_single_examples=True):
            if len(all_results) % 1000 == 0:
                tf.logging.info("Processing example: %d" % (len(all_results)))
            unique_id = int(result["unique_ids"])
            prediction = int(result["prediction"])
            all_results.append({unique_id: prediction})

        total = 0
        count = 0
        with tf.gfile.Open(dev_val_file, "r") as f:
            examples = json.load(f)
            # tf.logging.info(examples)
            for item in all_results:
                total += 1
                my_id = list(item.keys())[0]
                pr = int(item[my_id])
                gt = int(examples[str(my_id)])
                if pr == gt:
                    count += 1
        tf.logging.info("predict %d samples in total" % total)
        accuracy_score = float(count) / float(total)

        # result = estimator.evaluate(predict_input_fn)
        # accuracy_score = 0.0
        # for i in global_acc_list:
        #     accuracy_score += i
        # accuracy_score /= len(global_acc_list)
        tf.logging.info("Accuracy on dev set is %f" % accuracy_score)


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
