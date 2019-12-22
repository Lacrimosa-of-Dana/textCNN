import gensim
import tensorflow.compat.v1 as tf

from util import *


def test(path, batch_size=30):
    checkpoint_file = tf.train.latest_checkpoint('./runs/'+path+'/checkpoints/')
    data, y, word_dict, x = load_data('./resources/train.csv')
    embed_model = gensim.models.Word2Vec.load('./models/embedModel')
    tensor_model = embed_to_tensor(embed_model, word_dict)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_x = graph.get_operation_by_name('input_x').outputs[0]
            dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
            predictions = graph.get_operation_by_name('predictions').outputs[0]

            batches = batch_iter(list(x), batch_size, 1, shuffle=False)
            all_predictions = np.array([])
            for x_batch in batches:
                batch_predictions = sess.run(predictions,
                                             {input_x: x_batch,
                                              dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    correct_predictions = float(sum(all_predictions == np.argmax(y, 1)))
    print("Total number of test data: {}".format(len(y)))
    print("Accuracy: {:g}".format(correct_predictions / float(len(y))))

    return all_predictions


if __name__ == '__main__':
    test('1576997449')
