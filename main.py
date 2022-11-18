# Imports
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import unidecode
import contractions
import spacy
import time
import tkinter as tk
from tkinter import ttk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


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
                         # 'eatingdisorder': 'res/classification_data/datasets/eating_disorders.csv'
                         # 'bipolar': 'res/classification_data/datasets/bipolar.csv'
                         # 'ocd': res/classification_data/datasets/ocd.csv'
                         }

        df_list = []

        # Create the master-set
        for source, filepath in filepath_dict.items():
            df = pd.read_csv(filepath, names=['selftext'])

            # Cleanup / Optimize Master Dataset
            df = df[df.selftext.notnull()]  # Remove empty values
            df = df[df.selftext != '']  # Remove empty strings
            df = df[df.selftext != '[deleted]']  # Remove deleted status posts
            df = df[df.selftext != '[removed]']  # Remove removed status posts
            df['category'] = source  # Add category column
            df_list.append(df)
        df = pd.concat(df_list)

        # Extra data cleaning
        # Setup Spacy NLP and Customize Stopwords
        nlp = spacy.load('en_core_web_md')  # NLP Tools
        all_stopwords = nlp.Defaults.stop_words
        all_stopwords.remove('no')
        all_stopwords.remove('not')
        all_stopwords.add("/")
        all_stopwords.add('.')
        all_stopwords.add(",")
        all_stopwords.add("'")

        # print(all_stopwords) for debugging
        for i in range(0, len(df['selftext'])):
            # Get the value of current selftext
            value = df['selftext'].iloc[i]

            # Convert Accented Chars to standard chars
            value = unidecode.unidecode_expect_ascii(value)

            # Expand contractions
            value = contractions.fix(value)

            # Remove stopwords
            doc = nlp(value)
            text_tokens = [word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in doc]
            tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]

            value = tokens_without_sw

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
    os.makedirs(os.path.dirname(nb_filename), exist_ok=True)  # Create directory as needed
    try:
        print('Attempting to load nb.sav...')
        text_clf = joblib.load(nb_filename)
        print('Successfully Loaded nb.sav')
        return text_clf
    except FileNotFoundError:
        print('nb.sav not found. Setting up NB Classification Model.')
        print('Setting-Up Naive Bayes Classifier...')

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
        test_naive_bayes_classifier(text_clf, df)

        # Save model
        joblib.dump(text_clf, nb_filename)
        return text_clf


def classify_text(text_clf, input_text):
    print('Reached classify_text().')
    # Prepare classification output(s)
    output = text_clf.predict([input_text])
    classes = text_clf.classes_
    detailed_output = text_clf.predict_proba([input_text])

    # Store classification data into dictionary
    classification_dict = {}

    for i in range(0, len(classes)):
        classification_dict[classes[i]] = detailed_output[0][i]
    print(f'Prediction: {output}')
    print(f'Classification : {classification_dict}')

    return output


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

    # Add Notebook and frames to separate training mode and use
    notebook = ttk.Notebook(root)
    notebook.pack()

    # Setup Frames
    # Training Frame and Components
    training_frame = ttk.Frame(notebook, width=window_width, height=window_height)
    training_frame.pack(fill='both', expand=True)

    # Testing Frame and Components
    testing_frame = ttk.Frame(notebook, width=window_width, height=window_height)
    testing_frame.pack(fill='both', expand=True)

    # Output Label
    output_label = ttk.Label(testing_frame, text='Enter text to categorize.')
    output_label.pack(padx=10, pady=10)

    # Input Field
    input_field = ttk.Entry(testing_frame)
    input_field.pack(padx=10, pady=10)

    # Submit Button
    submit_button = ttk.Button(testing_frame,
                               text='Submit',
                               command=lambda: set_testing_output(classify_text(text_clf, input_field.get())))
    submit_button.pack(padx=10, pady=10)

    # Home Frame and Components
    home_frame = ttk.Frame(notebook, width=window_width, height=window_height)
    home_frame.pack(fill='both', expand=True)

    # Add frames to notebook
    notebook.add(training_frame, text='Training')
    notebook.add(testing_frame, text='Testing')
    notebook.add(home_frame, text='Home')
    notebook.select(testing_frame)  # Set default tab

    # Display main window and trigger focus
    print('Finished Building GUI.')
    root.focus()
    root.mainloop()


if __name__ == "__main__":
    main()
