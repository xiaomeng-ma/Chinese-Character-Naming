import os
import tensorflow as tf
import pandas as pd
import numpy as np
import collections
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer

def proc_df(args, df):
    df = df.astype({"tone": str})
    df['root'] = df['consonant'] + ',' + df['vowel']
    if args.pinyin == 'no':
        df['word'] = df['segment1'] + ',' + df['segment2']
    else:
        df['word'] = df['segment1'] + df['consonant1'] + df['vowel1'] + df['tone2'] + 'End' + df['segment2'] + df['consonant2'] + df['vowel2'] + df['tone2']
    if args.feature_spec == 'add_freq':
        df['word'] = ',Begin' + ',' + df['word'] + ',' + df['freq'] + ',' + 'End' 
    else:
        df['word'] = ',Begin' + ',' + df['word'] + ',' + 'End'

    if args.shuffle_spec == 'noshuffle':
        if args.tone_spec == 'notone':
            df['root'] = df['root'] + ',' + 'End'
        else:
            df['root'] = df['root'] + df['tone'] + ',' + 'End'
    else:
        if args.tone_spec == 'notone':
            df['root'] = df['vowel'] + ',' + df['consonant'] + ',' + 'End'
        else:
            df['root'] = df['vowel'] + ',' + df['consonant'] + ',' + df['tone'] + ',' + 'End'
    if args.label_spec == 'base':
        df['root'] = ',Begin,' + df['root']
    elif args.label_spec == 's':
        df['root'] = ',Begin,' + df['slabel'] + ',' + df['root']
    elif args.label_spec == 'sboth':
        df['root'] = ',Begin,' + df['slabel'] + ',' + df['slabel2'] + ',' + df['root']
    elif args.label_spec == 'm':
        df['root'] = ',Begin,' + df['mlabel'] + ',' + df['root']
    elif args.label_spec == 'mboth':
        df['root'] = ',Begin,' + df['mlabel'] + ',' + df['mlabel2'] + ',' + df['root']
    else:
        raise ValueError('Wrong label selected.')
    return df

def process_data(args):
    df_train = pd.read_csv(os.path.join(args.data_path, 'training.csv'), index_col=0)
    df_test = pd.read_csv(os.path.join(args.data_path, 'test.csv'), index_col=0)

    if args.freq_range == 'high':
        df_train = df_train[df_train['freq']=='high']
    elif args.freq_range == 'mid':
        df_train = df_train[(df_train['freq']=='high')|(df_train['frequency']=='mid')]
    elif args.freq_range == 'all':
        df_train = df_train
    df_train = proc_df(args, df_train)
    df_test = proc_df(args, df_test)

    dfall = df_train[['word', 'root', 'freq']]
    dftest = df_test[['word', 'root', 'freq']]
    dfall = dfall.rename(columns={'word': 'segment', 'root': 'character'})

    if args.feature_spec == 'add_freq':
        segment_train, segment_val, character_train, character_val, freq_train, freq_val = train_test_split(
            dfall['segment'].values, dfall['character'].values, dfall['frequency'].values, test_size=0.1,
            random_state=args.seed)
    else:
        segment_train, segment_val, character_train, character_val = train_test_split(
            dfall['segment'].values, dfall['character'].values, test_size=0.1, random_state=args.seed)

    t = Tokenizer(split=',',oov_token = '<unk>')
    # only use training set here, maybe we could keep those appear > 5.
    t.fit_on_texts(segment_train + character_train)
    # sorted(t.word_counts.items(), key=lambda item: item[1])
    # 159 appears only once, 576 words in total without label.
    # 81 words appear only once, 720 words in total without label.
    total_words = len(t.word_counts)
    rare_words = collections.Counter(t.word_counts[elem] for elem in t.word_counts)[1]
    print('total words {:d}, rare words {:d}'.format(total_words, rare_words))
    t.num_words = total_words 

    segment_seq_train = t.texts_to_sequences(segment_train)
    segment_seq_val = t.texts_to_sequences(segment_val)
    character_seq_train = t.texts_to_sequences(character_train)
    character_seq_val = t.texts_to_sequences(character_val)
    segment_seq_test = t.texts_to_sequences(dftest['word'].values)
    character_seq_test = t.texts_to_sequences(dftest['root'].values)

    train_examples = tf.data.Dataset.from_tensor_slices((tf.cast(np.array(segment_seq_train), tf.int64), tf.cast(np.array(character_seq_train), tf.int64)))
    val_examples = tf.data.Dataset.from_tensor_slices((tf.cast(np.array(segment_seq_val), tf.int64), tf.cast(np.array(character_seq_val), tf.int64)))
    test_examples = tf.data.Dataset.from_tensor_slices((tf.cast(np.array(segment_seq_test), tf.int64), tf.cast(np.array(character_seq_test), tf.int64)))
    BUFFER_SIZE = 60
    BATCH_SIZE = 16
    train_dataset = train_examples.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(16)
    val_dataset = val_examples.padded_batch(BATCH_SIZE)
    test_dataset = test_examples.padded_batch(60)
    return (train_dataset, val_dataset, test_dataset), t, (segment_seq_train, segment_seq_val, segment_seq_test)
