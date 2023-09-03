import numpy as np
import pandas as pd
from keras.src.preprocessing.text import Tokenizer


def preprocess_raw(in_file, out_csv):
    m2_file = open(in_file, "r").read().strip().split("\n\n")
    skippable_data = {"noop", "UNK", "Um"}

    correct = []
    incorrect = []

    for data in m2_file:
        corr_sentence = data.split("\n")[0].split(" ")[1:]
        edits = data.split("\n")[1:]
        offset = 0

        for edit in edits:
            edit = edit.split("|||")
            coder = int(edit[-1])

            if (edit[1] not in skippable_data) and (coder == 0):
                edit_data = edit[0].split()[1:]
                start = int(edit_data[0])
                end = int(edit_data[1])
                corr = edit[2].split()
                corr_sentence[start + offset: end + offset] = corr
                offset = offset - (end - start) + len(corr)
        correct.append(" ".join(corr_sentence))

    for data in m2_file:
        temp = data.split("\n")[0].split(" ")[1:]
        temp = " ".join(temp)
        incorrect.append(temp)

    df = pd.DataFrame()

    df["correct"] = correct
    df["incorrect"] = incorrect

    sameSent = []

    for i in range(len(df.values)):
        if df.values[i][0] == df.values[i][1]:
            sameSent.append(i)

    df.drop(sameSent, inplace=True)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)

    df.to_csv(out_csv, index=False)

    return df


def tokenize_and_pad(df):
    df["correct_inp"] = "$" + df["correct"].astype(str)
    df["correct_out"] = df["correct"].astype(str) + "@"

    tokenizer_inc = Tokenizer(filters="", lower=True, char_level=False)
    tokenizer_inc.fit_on_texts(df["incorrect"].values)

    inc_train = np.array(tokenizer_inc.texts_to_sequences(df["incorrect"].values))
    print(inc_train)

    return df


training_df = preprocess_raw("test.m2", "data_test.csv")
print("Done preprocessing!")

tokenize_and_pad(training_df)