# Imports
import threading

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import unidecode
import contractions
import re
import spacy
import time
import plotly.express as px
import tkinter as tk
import speech_recognition as sr
from tkinter import ttk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Setup Spacy NLP and Customize Stopwords
print('Loading Spacy NLP...')

nlp = spacy.load('en_core_web_md')  # NLP Tools
all_stopwords = nlp.Defaults.stop_words
all_stopwords.remove('no')
all_stopwords.remove('not')
all_stopwords.remove('out')
all_stopwords.remove('empty')
all_stopwords.remove('alone')
all_stopwords.remove('myself')
all_stopwords.add("/")
all_stopwords.add('.')
all_stopwords.add(",")
all_stopwords.add("'")

print('Spacy NLP has Loaded Successfully.')
print(all_stopwords)


def text_cleanup(text):
    # Extra data cleaning
    # Make text lowercase
    text = text.lower()
    # Convert Accented Chars to standard chars
    text = unidecode.unidecode_expect_ascii(text)

    # Remove links from text
    text = re.sub(r'http\S+', '', text)

    # Remove \r and \n and parenthesis from string
    text = text.replace('\r', '')
    text = text.replace('\n', '')
    text = text.replace('(', '')
    text = text.replace(')', '')

    # Remove reddit status text
    text = text.replace('view poll', '')
    text = text.replace('deleted', '')
    text = text.replace('[removed]', '')

    # Remove numbers from string
    text = re.sub(r'[0-9]+', '', text)

    # Expand contractions
    text = contractions.fix(text)

    # Remove stopwords
    doc = nlp(text)
    text_tokens = [word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in doc]
    tokens_wo_stopwords = [word for word in text_tokens if not word in all_stopwords]
    cleaned_text = ' '.join(tokens_wo_stopwords)

    return cleaned_text


def load_data():
    # Configure Filepaths
    master_filepath = 'res/classification_data/datasets/master-set.csv'

    try:  # Try to load existing master dataset. If not found, create new one
        return pd.read_csv(master_filepath)
    except FileNotFoundError:

        filepath_dict = {'anxiety': 'res/classification_data/datasets/anxiety.csv',
                         'depression': 'res/classification_data/datasets/depression.csv',
                         'tourettes': 'res/classification_data/datasets/tourettes.csv',
                         'suicide': 'res/classification_data/datasets/suicidewatch.csv',
                         'adhd': 'res/classification_data/datasets/adhd.csv',
                         'schizophrenia': 'res/classification_data/datasets/schizophrenia.csv',
                         'eatingdisorder': 'res/classification_data/datasets/eatingdisorder.csv',
                         'bipolar': 'res/classification_data/datasets/bipolar.csv',
                         'ocd': 'res/classification_data/datasets/ocd.csv'
                         }

        df_list = []

        # Create the master-set
        for source, filepath in filepath_dict.items():
            # Load the selftext columns of each file
            df = pd.read_csv(filepath, names=['selftext'])

            # Cleanup / Optimize Master Dataset
            df = df[df.selftext.notnull()]  # Remove empty values
            df = df[df.selftext != '']  # Remove empty strings
            df = df[df.selftext != '[deleted]']  # Remove deleted status posts
            df = df[df.selftext != '[removed]']  # Remove removed status posts
            df['category'] = source  # Add category column
            df_list.append(df)
        df = pd.concat(df_list)

        for i in range(0, len(df['selftext'])):
            # Get the value of current selftext
            value = df['selftext'].iloc[i]

            # Clean the data
            value = text_cleanup(value)

            # Update the dataframe for master-set
            df['selftext'].iloc[i] = value
            print(df['selftext'].iloc[i])

        print(f'5 Samples: {df.head()}\n| Summary: \n{df.info}\nDescription: {df.describe()}\nShape: {df.shape}')

        # Make master-set csv and save .csv file
        df.to_csv('res/classification_data/datasets/master-set.csv', index=0)
        print('Master Dataset Created.')

        return df


def plot_training_results(pass_score_dict, fail_score_dict):
    print('Reached plot_training_results()')

    # Plot performance
    plt.rcParams['figure.figsize'] = [7.5, 3.5]
    plt.rcParams['figure.autolayout'] = True

    # Pass Performance
    pass_score_dict = np.array(pass_score_dict)
    x = np.arange(0, len(pass_score_dict))
    y = pass_score_dict
    plt.plot(x, y, color="blue", label="Pass")

    # Fail performance
    fail_score_dict = np.array(fail_score_dict)
    x_fail = np.arange(0, len(fail_score_dict))
    y_fail = fail_score_dict
    plt.plot(x_fail, y_fail, color="red", label="Fail")

    # Customize Scatter Plot
    plt.title("NB Classifier Accuracy")
    plt.xlabel("Number of Samples")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.show()  # Show the scatter plot


def test_naive_bayes_classifier(text_clf, df):
    print('Reached test_naive_bayes_classifier()')
    print(df.category[0])
    # Testing accuracy and populate dicts to use to plot
    i, pass_count, fail_count = 0, 0, 0
    pass_score_dict = []
    fail_score_dict = []
    test_list = []
    shuffled_df = df.sample(frac=.2)  # Use 20% of data as testing data
    start_time = time.time()

    # Iterative Performance Measuring of NB Classifier
    print('Analyzing Classifier Performance...')
    for selftext in shuffled_df.selftext:
        # Make prediction, get actual value, and get current time elapsed
        pred = text_clf.predict([selftext])
        actual = shuffled_df['category'].values[i]
        time_elapsed = (time.time() - start_time)

        # Populate pass/fail lists
        if pred == actual:
            pass_count = pass_count + 1
        else:
            fail_count = fail_count + 1

        # Update pass/fail score
        pass_score = pass_count / len(shuffled_df)
        pass_score_dict.append(pass_score)

        fail_score = fail_count / len(shuffled_df)
        fail_score_dict.append(fail_score)

        # Populate test_list and print detailed for monitoring
        test_result = f'ID: {i + 1}/{len(shuffled_df)} | Pass Score: {pass_score} ' \
                      f'| Fail Score: {fail_score} | Prediction: {pred} ' \
                      f'| Actual:{actual} | SelfText: {selftext}'
        print(test_result)
        print(f'Time Elapsed: {time_elapsed:.2f}m | {test_result}')

        # Update test_list
        test_list.append(test_result)

        # Increment the index
        i = i + 1

    # Save test results to .csv
    test_df = pd.DataFrame(test_list, columns=['Test Results'])
    test_df.to_csv('res/classification_data/datasets/test_results.csv', index=0)
    print("Detailed Testing Complete - test_results.csv created.")

    # General Performance Measuring of NB Classifier
    predicted = text_clf.predict(df.selftext)
    score = np.mean(predicted == df.category)
    print(f'Performance Analysis Completed in {(time.time() - start_time) / 60} minutes.')
    print(f'Average Performance (Naive Bayes): {score:.3f}%')

    # Plot the performance of the NB Classifier
    plot_training_results(pass_score_dict, fail_score_dict)


def naive_bayes_classifier(df):
    # Attempt to load existing model. If model isn't found, create a new one
    nb_filename = 'res/classification_data/models/nb.sav'
    try:
        print('Attempting to load nb.sav...')
        text_clf = joblib.load(nb_filename)
        print('Successfully Loaded nb.sav')
        return text_clf
    except FileNotFoundError:
        print('nb.sav not found. Setting up NB Classification Model.')
        print('Setting-Up Naive Bayes Classifier...')

        if not os.path.exists('res/classification_data/models'):
            os.makedirs('res/classification_data/models')

        # Setup NB Classification Pipeline
        text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB())])
        print("Features Extracted.")
        print("Term Frequencies Extracted.")
        print('Naive Bayes Classifier Setup Complete.')

        # Run Naive Bayes(NB) ML Algorithm
        text_clf = text_clf.fit(df.selftext, df.category)

        # Test Performance of NB Classifier
        # test_naive_bayes_classifier(text_clf, df)

        # Save model
        joblib.dump(text_clf, nb_filename)
        return text_clf


def plot_bubble_chart(classification_df):
    label = [i + '<br>' + str(j) + '%' for i, j in zip(classification_df.category,
                                                       classification_df.probability)]
    # label = 'label'
    fig = px.scatter(classification_df, x='X', y='Y',
                     color='category',
                     size='probability', text=label, size_max=90)

    fig.update_layout(width=900, height=320,
                      margin=dict(t=50, l=0, r=0, b=0),
                      showlegend=True
                      )
    fig.update_traces(textposition='top center')
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_layout({'plot_bgcolor': 'white',
                       'paper_bgcolor': 'white'})
    fig.show()


def classify_text(text_clf, input_text):
    print('Reached classify_text().')
    if input_text == '':
        return ''
    # Clean the input text
    print('Cleaning Input Text...')
    cleaned_text = text_cleanup(input_text)
    print(f'Cleaned Text: {cleaned_text}')
    print('Finished Cleaning Input Text.\nClassifying Text...')

    # Prepare classification output(s)
    output = text_clf.predict([cleaned_text])
    classes = text_clf.classes_.tolist()

    # Shorten decimals for percentages
    detailed_output = text_clf.predict_proba([cleaned_text]).tolist()
    detailed_output = detailed_output[0]
    for i in range(0, len(detailed_output)):
        detailed_output[i] = (round(detailed_output[i] * 100, 2))

    print(f'Classes: {classes}')
    print(f'Detailed Output: {detailed_output}')

    # Store classification data into dataframe
    data = {
        "category": classes,
        "probability": detailed_output,
        "X": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "Y": [1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    cdf = pd.DataFrame(data)

    print(cdf)
    # plot_bubble_chart(cdf)
    print('Classification Complete.')

    return output


def listening(text_clf):
    recognizer = sr.Recognizer()
    whole_input = ""
    exit_phrase = 'end session'
    print("Listening for input...")
    while True:
        with sr.Microphone(0) as mic:
            audio = recognizer.record(mic, duration=5)
            try:
                text = recognizer.recognize_google(audio)
                if exit_phrase in text.lower():
                    break
                whole_input = whole_input + " " + text
            except Exception as e:
                whole_input = whole_input
                print("Error: No input added.")
            print(f'Input: {whole_input}')
    classify_text(text_clf, whole_input)


def speech_to_text(text_clf):
    stt_thread = threading.Thread(target=listening(text_clf))
    stt_thread.start()


def main():
    def set_testing_output(output):
        output_label.config(text=output)

    print("Reached main().")
    # Load and consolidate the datasets
    df = load_data()
    print("Loaded dataframe.")

    # Run Naive Bayes(NB) Machine-learning Algorithm
    text_clf = naive_bayes_classifier(df)

    # Build GUI
    print('Building GUI...')
    root = tk.Tk()
    root.title("ESAI: Your Emotional Support AI")

    # Setup window dimensions
    window_width = 720
    window_height = 480

    # Get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Find the center point
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)

    # Configure Window
    root.geometry(f'{400}x{400}+{center_x + 160}+{center_y}')
    root.resizable(width=False, height=False)  # Prevent Resizing

    # Output Label
    output_label = ttk.Label(root, text='Enter text to categorize.')
    output_label.pack(padx=10, pady=10)

    # Input Field
    input_field = ttk.Entry(root)
    input_field.focus()
    input_field.pack(padx=10, pady=10)

    # Submit Button
    submit_button = ttk.Button(root,
                               text='Submit',
                               command=lambda: set_testing_output(classify_text(text_clf, input_field.get())))
    submit_button.pack(side='left', padx=10, pady=10)

    # STT Button
    microphone_icon = tk.PhotoImage(file="res/img/microphone_icon.png")
    stt_button = ttk.Button(root, image=microphone_icon, width=10, command=lambda: speech_to_text(text_clf))
    stt_button.pack(side="right", padx=5, pady=10)

    # Display main window and trigger focus
    print('Finished Building GUI.')
    root.mainloop()


if __name__ == "__main__":
    main()
