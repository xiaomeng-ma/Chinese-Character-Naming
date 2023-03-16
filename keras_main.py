from tensorflow.python.client import device_lib
import os
import sys
import time
import logging
import data_process
import config
import pandas as pd
from models import *

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp],
                                     training=True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))

def dev_step(val_dataset):
    corr_num, p_corr_num, total_num, p_total_num = 0., 0., 0., 0.
    output_list = []
    start, end = 1, -1
    p_start, p_end = 1, -1
    if args.tone_spec == 'notone':
        p_end -= 1
    if args.label_spec == 's' or args.label_spec == 'm':
        p_start += 1
        start += 1
    elif args.label_spec == 'sboth' or args.label_spec == 'mboth':
        p_start += 2
        start += 2
    for (batch, (inp, tar)) in enumerate(val_dataset):
        output = tar[:, :1]
        bsz, seq_len = float(tf.shape(tar)[0]), float(tf.shape(tar)[1])
        for i in tf.range(seq_len - 1):
            pred, _ = transformer([inp, output], training=False)

            # select the last token from the seq_len dimension
            predictions = pred[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat(axis=1, values=[output, predicted_id])
        output_list.append(output)
        corr_num += float(tf.reduce_sum(tf.cast(tf.equal(tar[:, start:end], output[:, start:end]), tf.float32)))
        p_corr_num += float(tf.reduce_sum(tf.cast(tf.equal(tar[:, p_start:p_end], output[:, p_start:p_end]), tf.float32)))
        total_num += (seq_len - start + end) * bsz
        p_total_num += (seq_len - p_start + p_end) * bsz
    return corr_num / total_num, p_corr_num / p_total_num, output_list


if __name__ == "__main__":
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    args = config.get_args()
    args = config.process_args(args)

    # fix random seeds
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # set logging settings
    from importlib import reload

    reload(logging)
    log_file = os.path.join(args.model_path, 'log')
    handlers = [logging.FileHandler(log_file, mode='w+'), logging.StreamHandler()]
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M', level=logging.INFO, handlers=handlers)
    logging.info(args)

    # check cpu or gpu
    logging.info(device_lib.list_local_devices())

    # read data
    dataset, t, segment_seq = data_process.process_data(args)
    train_dataset, val_dataset, test_dataset = dataset
    segment_seq_train, segment_seq_val, segment_seq_test = segment_seq
    num_batches, val_batches = len(train_dataset), len(val_dataset)

    # settings
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    metrics = [accuracy]
    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)
    optimizer = tf.keras.optimizers.Adam(CustomSchedule(args.d_model), beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    # 15 epochs for all,
    EPOCHS = 50

    # create model
    vocab_num = t.num_words
    transformer = Transformer(
        num_layers=args.nlayers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        input_vocab_size=vocab_num,
        target_vocab_size=vocab_num,
        rate=args.dropout)
    checkpoint_path = args.model_path
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
    p_ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(checkpoint_path, 'no_tone'), max_to_keep=1)
    # if a checkpoint exists, stop and logging.info warning
    if ckpt_manager.latest_checkpoint:
        logging.info('this is already trained')
        # sys.exit(0)
        epoch, best_epoch, best_p_epoch = 0., 0., 1.
    else:
        # start training
        best_model, best_dev_acc, best_epoch = None, 0.0, 0.0
        best_p_model, best_p_dev_acc, best_p_epoch = None, 0.0, 0.0
        for epoch in range(EPOCHS):
            start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            # inp -> input, tar -> target
            for (batch, (inp, tar)) in enumerate(train_dataset):
                train_step(inp, tar)
            logging.info(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

            dev_acc, p_dev_acc, _ = dev_step(val_dataset)
            logging.info(f'Epoch {epoch + 1} Dev Accuracy {dev_acc:.4f}, Dev Accuracy with pinyin {p_dev_acc:.4f}')
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_epoch = epoch + 1
                ckpt_save_path = ckpt_manager.save()
                logging.info(f'Saving best model for epoch {epoch + 1} at {ckpt_save_path}')

            if p_dev_acc > best_p_dev_acc:
                best_p_dev_acc = p_dev_acc
                best_p_epoch = epoch + 1
                ckpt_save_path = p_ckpt_manager.save()
                logging.info(f'Saving best pinyin model for epoch {epoch + 1} at {ckpt_save_path}')

            logging.info(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
    # load model and test
    # checkpoint_fname = checkpoint_path + '/ckpt-' + str(15)
    # tf.print("-----------Restoring from {}-----------".format(checkpoint_fname))
    # ckpt.restore(checkpoint_fname)

    output_acc = []
    for idx, ckpt_load in enumerate([ckpt_manager, p_ckpt_manager]):
        if ckpt_load.latest_checkpoint:
            ckpt.restore(ckpt_load.latest_checkpoint)
            logging.info('Latest checkpoint restored!!')
        # tf.keras.models.save_model(transformer, './saved_model/model', include_optimizer=False)
        # model = tf.keras.models.load_model('./saved_model/model')

        # test
        if epoch == 0:
            best_dev_acc, best_p_dev_acc, _ = dev_step(val_dataset)
        test_acc, p_test_acc, pred_list = dev_step(test_dataset)
        pred_list = pred_list[0]
        logging.info(f'Epoch {epoch + 1} Test Accuracy {test_acc:.4f}, Pinyin Test Accuracy {p_test_acc:.4f}')
        if idx == 0:
            output_acc += [best_dev_acc * 100, test_acc * 100]
        else:
            output_acc += [best_p_dev_acc * 100, p_test_acc * 100]
        df_test = pd.read_csv(os.path.join(args.data_path, 'test.csv'), index_col=0
                              ).astype({"tone": str})
        df_test['root'] = ',Begin' + ',' + df_test['consonant'] + ',' + \
                          df_test['vowel'] + ',' + df_test['tone'] + ',' + 'End'
        root_list = []
        for xx in df_test['root'].values:
            root_list.append(",".join(xx.split(',')[2:5]))
        df_test['root'] = root_list
        df_test['pred'] = pd.Series(pred_list.numpy().tolist(),
                                    index=df_test.index[:len(pred_list)])
        df_test['pred'] = df_test['pred'].apply(lambda row: t.sequences_to_texts([row])[0])
        df_test[['root', 'pred']].to_csv(os.path.join(ckpt_load.directory, 'results.csv'))

        new = df_test['pred'].str.split(' ', expand=True)
        df_test = df_test[['consonant', 'vowel', 'tone']]

        start = 1
        if args.label_spec == 'sboth' or args.label_spec == 'mboth':
            start += 2
        elif args.label_spec == 's' or args.label_spec == 'm':
            start += 1
        if args.shuffle_spec == 'shuffle':
            df_test['consonant_pred'] = new[start + 1]
            df_test['vowel_pred'] = new[start]
        else:
            df_test['consonant_pred'] = new[start]
            df_test['vowel_pred'] = new[start + 1]
        test_size = df_test.shape[0]

        df_test = df_test.reset_index()
        consonant_corr = df_test.loc[df_test.consonant == df_test.consonant_pred]
        vowel_corr = df_test.loc[df_test.vowel == df_test.vowel_pred]
        consonant_acc = consonant_corr.shape[0] / test_size * 100
        vowel_acc = vowel_corr.shape[0] / test_size * 100
        logging.info('consonant acc: {:.2f}%, correct num: {:d}'.format(consonant_acc, consonant_corr.shape[0]))
        logging.info('vowel acc: {:.2f}%, correct num: {:d}'.format(vowel_acc, vowel_corr.shape[0]))
        if args.tone_spec == 'tone':
            df_test['tone_pred'] = new[start + 2]
            tone_corr = df_test.loc[df_test.tone == df_test.tone_pred]
            tone_acc = tone_corr.shape[0] / test_size * 100
            logging.info('tone acc: {:.2f}%, correct num: {:d}'.format(tone_acc, tone_corr.shape[0]))
        else:
            tone_acc = 0.
        output_acc += [consonant_acc, vowel_acc, tone_acc]

        s1 = pd.merge(consonant_corr, vowel_corr, how='inner')
        logging.info('p test acc: {:.2f}%, correct num: {:d}'.format(s1.shape[0] / test_size * 100, s1.shape[0]))
        if args.tone_spec == 'tone':
            s1 = pd.merge(s1, tone_corr, how='inner')
            logging.info('test acc with tone: {:.2f}%, correct num: {:d}'.format(s1.shape[0] / test_size * 100,
                                                                                 s1.shape[0]))
    df_final = pd.DataFrame([output_acc], columns=['dev_acc', 'test_acc', 'consonant_acc', 'vowel_acc', 'tone_acc',
                                                   'p_dev_acc', 'p_test_acc', 'p_consonant_acc',
                                                   'p_vowel_acc', 'p_tone_acc'])
    df_final.to_csv(os.path.join(args.model_path, 'dev_test_acc.csv'))

