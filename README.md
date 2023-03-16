# Chinese-Character-Naming

Hi! This is the companion repository for: **Xiaomeng Ma** and **Lingyu Gao**. Evaluating Transformer Models and Human Behaviors on Chinese Character Naming. To Appear at *TACL*.

If you have any questions and comments, feel free to reach out to Xiaomeng Ma: xm2158@tc.columbia.edu.

# Human answers

The [human_answer.csv](https://github.com/xiaomeng-ma/Chinese-Character-Naming/blob/main/human_answer.csv) listed the answers of 55 participants for the 60 characters. 

# Experiment

(run the following command)

`python keras_main.py -seed XX -pinyin XX -label_spec XX -tone_spec XX -shuffle_spec XX -freq_range XX -feature_spec XX`

| Parameters    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |   |
|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| -pinyin       | no: no radical's pinyin in the input (Model[-pinyin] in the paper) add: add radical's pinyin in the input (Model[+pinyin] in the paper)                                                                                                                                                                                                                                                                                                                                                  |   |
| -label_spec   | base: output no label, only the pinyin of the character (base model in the paper) s: output the radical's position label based on phonetic similarity (label_s in the paper) m: output the radical's position label based on dictionary (label_m in the paper) sboth: output both phonetic similarity position label and regularity based on that label (label_sr in the paper) mboth: output both dictionary position label and regularity based on that label (label_mr in the paper)  |   |
| -tone_spec    | notone: output the character's pinyin without tone (-Tone in the paper) tone:output the character's pinyin with tone (+Tone in the paper)                                                                                                                                                                                                                                                                                                                                                |   |
| -shuffle_spec | noshuffle: don't shuffle the consonant and vowel in the output (-Shuffle in the paper) shuffle: shuffle the consonant and vowel in the output (+Shuffle in the paper)                                                                                                                                                                                                                                                                                                                    |   |
| -freq_range   | all: use all characters as input (all in the paper) mid: use characters with high + mid frequency label as input (mid in the paper) high: use characters with only with high frequency label as input (high in the paper)                                                                                                                                                                                                                                                                |   |
| -feature_spec | no: don't add frequency label in the input add_freq: add frequency label in the input (only applied to all characters, all+freq in the paper)                                                                                                                                                                                                                                                                                                                                            |   |

Seeds used in the paper: 42, 1, 11, 111, 1111

Epochs: 40 with early stopping

Batch size: 16

