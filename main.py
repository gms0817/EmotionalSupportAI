# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import tkinter as tk
from tkinter import ttk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def load_data():
    # Configure Filepaths
    filepath_dict = {'anxiety': 'res/classification_data/datasets/anxiety.csv',
                     'depression': 'res/classification_data/datasets/depression.csv',
                     'tourettes': 'res/classification_data/datasets/tourettes.csv',
                     'suicide': 'res/classification_data/datasets/suicidewatch.csv'}

    df_list = []

    # Create the master-set
    for source, filepath in filepath_dict.items():
        df = pd.read_csv(filepath, names=['selftext'])
        df = df[df.selftext.notnull()]  # Remove empty values
        df = df[df.selftext != '']  # Remove empty strings
        df = df[df.selftext != '[deleted]']  # Remove deleted status posts
        df = df[df.selftext != '[removed]']  # Remove removed status posts
        df['category'] = source  # Add category column
        df_list.append(df)

    df = pd.concat(df_list)
    # print(f'5 Samples: {df.head()}\n| Summary: \n{df.info}\nDescription: {df.describe()}\nShape: {df.shape}')

    # Make master-set csv and save .csv file
    df.to_csv('res/classification_data/datasets/master-set.csv', index=0)
    print('Master Dataset Created.')
    return df


# Use this function to generate a detailed csv report of NB Classification Testing Result
def detailed_naive_bayes_classifier(df):
    # Extract features from files based on the 'bag-of-words' model
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())])
    print("Features Extracted.")
    print("Term Frequencies Extracted.")

    # Run Naive Bayes(NB) ML Algorithm
    text_clf = text_clf.fit(df.selftext, df.category)

    # Test Performance of NB Classifier (Detailed)
    i = 0
    test_list = []
    shuffled_df = df.sample(frac=20)  # Get a random sample to use so that each illness/disorder is tested
    start_time = time.time()
    for selftext in shuffled_df.selftext:
        pred = text_clf.predict(shuffled_df.selftext)
        actual = shuffled_df.category[i]
        time_elapsed = (time.time() - start_time) / 60
        if pred[i] == actual:
            result = 'PASS'
        else:
            result = 'FAIL'
        test = f'ID: {i + 1}/{len(shuffled_df)} | Prediction: {pred[i]} | Actual: {actual} ' \
               f'| Result: {result} | Selftext: {selftext}'
        print(f'Time Elapsed: {time_elapsed:.2f}m | {test}')
        i = i + 1

        # Store Test into list
        test_list.append(test)

    total_time = (time.time() - start_time) / 60
    # Save test results to .csv
    test_df = pd.DataFrame(test_list, columns=['Results'])
    test_df.to_csv('classification_data/datasets/test_results.csv', index=0)
    print("Detailed Testing Complete - test_results.csv created.")
    print(f"Total Time Elaped: {total_time:.2f}m")


def naive_bayes_classifier(df):
    print('Setting-Up Naive Bayes Classifier...')
    # Extract features from files based on the 'bag-of-words' model
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())])
    print("Features Extracted.")
    print("Term Frequencies Extracted.")
    print('Naive Bayes Classifier Setup Complete.')

    # Run Naive Bayes(NB) ML Algorithm
    text_clf = text_clf.fit(df.selftext, df.category)

    # Testing accuracy and populate dicts to use to plot
    i, pass_count, fail_count = 0, 0, 0
    pass_score_dict = []
    fail_score_dict = []

    print('Analyzing Classifier Performance...')
    # Manual Performance Measuring of NB Classifier
    for selftext in df.selftext:
        pred = text_clf.predict([selftext])
        actual = df.category[i]
        # print(f'Prediction: {pred} | Actual:{actual} | SelfText: {selftext}')

        # Populate pass/fail lists
        if pred == actual:
            pass_count = pass_count + 1
        else:
            fail_count = fail_count + 1
        pass_score = pass_count / len(df)
        pass_score_dict.append(pass_score)

        fail_score = fail_count / len(df)
        fail_score_dict.append(fail_score)

        print(f'Pass Score: {pass_score} | Fail Score: {fail_score}')

        i = i + 1

    # General Performance Measuring of NB Classifier
    predicted = text_clf.predict(df.selftext)
    score = np.mean(predicted == df.category)
    print(f'Performance Analysis Complete.\nAverage Performance (Naive Bayes): {score:.3f}%')

    # Uncomment to produce plot of performance
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

    return text_clf


def classify_text(text_clf, input_text):
    print('Reached classify_text().')
    output = text_clf.predict([input_text])

    print(f'Classification: {output}')
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
    root.resizable(width=False, height=False) # Prevent Resizing

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
    notebook.select(testing_frame) # Set default tab

    # Display main window and trigger focus
    print('Finished Building GUI.')
    root.focus()
    root.mainloop()


if __name__ == "__main__":
    main()
