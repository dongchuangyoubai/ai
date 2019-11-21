import os
import argparse
import tensorflow as tf
from auto_encoder import AutoEncoder
from language_model import LanguageModel
from data_utils import build_word_dict, build_word_dataset, batch_iter, download_dbpedia

BATCH_SIZE = 64
NUM_EPOCHS = 10
MAX_DOCUMENT_LEN = 100


def train(train_x, train_y, word_dict, args):
    with tf.compat.v1.Session() as sess:
        # model = AutoEncoder(word_dict, MAX_DOCUMENT_LEN)
        model = LanguageModel(word_dict, MAX_DOCUMENT_LEN)
        global_steps = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(model.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

        loss_summary = tf.summary.scalar("loss", model.loss)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("AutoEncoder", sess.graph)

        saver = tf.train.Saver(tf.global_variables())

        sess.run(tf.global_variables_initializer())

        def train_step(batch_x):
            feed_dict = {model.x: batch_x}
            _, step, summaries, loss = sess.run([train_op, global_steps, summary_op, model.loss],
                                                feed_dict=feed_dict)
            summary_writer.add_summary(summaries, step)

            if step % 100 == 0:
                print("step {0} : loss = {1}".format(step, loss))

        batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)

        for batch_x, _ in batches:
            train_step(batch_x)
            step = tf.train.global_step(sess, global_steps)

            if step % 5000 == 0:
                saver.save(sess, os.path.join("AutoEncoder", "model", "model.ckpt"), global_step=step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="auto_encoder", help="auto_encoder | language_model")
    args = parser.parse_args()
    # import nltk
    # nltk.download('punkt')

    if not os.path.exists("dbpedia_csv"):
        print("Downloading dbpedia dataset...")
        download_dbpedia()

    print("\nBuilding dictionary..")
    word_dict = build_word_dict()

    train_x, train_y = build_word_dataset("debug", word_dict, MAX_DOCUMENT_LEN)
    train(train_x, train_y, word_dict, args)