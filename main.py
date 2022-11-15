# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def load_data():
    # Configure Filepaths
    filepath_dict = {'anxiety': 'res/classification_data/datasets/anxiety.csv',
                     'depression': 'res/classification_data/datasets/depression.csv',
                     'tourettes': 'res/classification_data/datasets/tourettes.csv'}

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
    print(f'5 Samples: {df.head()}\n| Summary: \n{df.info}\nDescription: {df.describe()}\nShape: {df.shape}')

    # Make master-set csv and save .csv file
    df.to_csv('res/classification_data/datasets/master-set.csv', index=0)

    return df


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
    # Extract features from files based on the 'bag-of-words' model
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())])
    print("Features Extracted.")
    print("Term Frequencies Extracted.")

    # Run Naive Bayes(NB) ML Algorithm
    text_clf = text_clf.fit(df.selftext, df.category)

    # Testing accuracy and populate dicts to use to plot
    i, pass_count, fail_count = 0, 0, 0
    pass_score_dict = []
    fail_score_dict = []

    # Manual Performance Measuring of NB Classifier
    for selftext in df.selftext:
        pred = text_clf.predict([selftext])
        actual = df.category[i]
        print(f'Prediction: {pred} | Actual:{actual} | SelfText: {selftext}')

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

    plt.show() # Show the scatter plot

    # General Performance Measuring of NB Classifier
    predicted = text_clf.predict(df.selftext)
    print(f'Predicted: {predicted}')
    score = np.mean(predicted == df.category)
    print(f'Average Performance (Naive Bayes): {score:.3f}%')
    print("General Testing Complete.")

    return text_clf


def main():
    print("Reached main().")
    # Load and consolidate the datasets
    df = load_data()
    print("Loaded dataframe.")

    # Run Naive Bayes(NB) Machine-learning Algorithm
    text_clf = naive_bayes_classifier(df)

    test1 = "I have been feeling really sad recently and i am not sure what to do"
    pred1 = text_clf.predict([test1])

    test2 = "I'm so stressed and anxious all the time and i dont know whats going on"
    pred2 = text_clf.predict([test2])
    print(f'Prediction: {pred1} | Test 1: {test1}')
    print(f'Prediction: {pred2} | Test 2: {test2}')


if __name__ == "__main__":
    main()