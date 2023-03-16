# Imports
import csv
import datetime
import pickle
import queue

import matplotlib.pyplot as plt
import numpy as np
import docx  # Install python-docx not docx for python 3.9
import pyttsx3
import spacy
import time
import sv_ttk
import threading
import joblib
import os
import unidecode
import contractions
import re
import wave
import pandas as pd
import tkinter as tk
import speech_recognition as sr
from tkinter import ttk, filedialog
from tkinter import *

from matplotlib.backends._backend_tk import NavigationToolbar2Tk
import pyaudio
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def text_cleanup(text):
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


def plot_training_results(pass_score_dict, fail_score_dict, classifier):
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
    plt.title(f'{classifier} Accuracy')
    plt.xlabel("Number of Data Samples")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.show()  # Show the scatter plot


def test_classifier(model, df, classifier, dataName):
    print('Reached test_classifier()')

    # Testing accuracy and populate dicts to use to plot
    df = df.fillna('')

    i, pass_count, fail_count = 0, 0, 0
    pass_score_dict = []
    fail_score_dict = []
    test_list = []
    shuffled_df = df.sample(frac=.2)  # Use 20% of data as testing data
    start_time = time.time()

    # Iterative Performance Measuring of NB Classifier
    print('Analyzing Classifier Performance...')
    for element in shuffled_df[dataName]:
        # Make prediction, get actual value, and get current time elapsed
        pred = model.predict([element])
        if dataName == 'selftext':  # MHA Classifier
            actual = shuffled_df['category'].values[i]
        else:  # SaiBot Classifier
            actual = shuffled_df['response'].values[i]
        time_elapsed = (time.time() - start_time) / 60

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
        test_result = f'Classifier: {classifier} | ' \
                      f'ID: {i + 1}/{len(shuffled_df)} | Pass Score: {pass_score} ' \
                      f'| Fail Score: {fail_score} | Prediction: {pred} ' \
                      f'| Actual:{actual} | {dataName}: {element}'
        print(test_result)
        print(f'Time Elapsed: {time_elapsed:.2f}m | {test_result}')

        # Update test_list
        test_list.append(test_result)

        # Increment the index
        i = i + 1

    # Save test results to .csv
    test_df = pd.DataFrame(test_list, columns=['Test Results'])
    test_df.to_csv(f'res/classification_data/datasets/{classifier}-test_results.csv', index=0)
    print("Detailed Testing Complete - test_results.csv created.")

    # General Performance Measuring of NB Classifier
    if dataName == 'selftext':  # MHA Classifier
        predicted = model.predict(df[dataName])
        score = np.mean(predicted == df['category'])
    else:  # SaiBot Classifier
        predicted = model.predict(df[dataName])
        score = np.mean(predicted == df['response'])
    time_elapsed = (time.time() - start_time) / 60

    print(f'Performance Analysis Completed in {time_elapsed:.2f} minutes.')
    print(f'Average Performance (Naive Bayes): {score:.2f}%')

    # Plot the performance of the NB Classifier
    plot_training_results(pass_score_dict, fail_score_dict, classifier)


class Journal:
    def __init__(self):
        self.entryList = []

    # Add journal entry
    def addEntry(self, journalEntry):
        print('Reached addEntry()')
        self.entryList.append(journalEntry)

    # Load Journal File
    def loadJournal(self):
        # Create journal if it doesn't exist
        currentJournalDate = datetime.datetime.now().strftime('%m-%Y')
        path = r'UserData/journals/%s.obj' % currentJournalDate
        try:
            filehandler = open(path, 'rb')
            self.entryList = pickle.load(filehandler)
            print(f'Successfully loaded journal for {currentJournalDate}')
        except FileNotFoundError:
            print('journal.obj not found. Creating new journal')

            # Dump the file
            file = open(path, 'wb')
            pickle.dump(self.entryList, file)

    # Export journal file
    def exportJournal(self):
        print('Reached exportJournal()')
        file = open('filename_pi.obj', 'wb')
        pickle.dump(self.entryList, file)


class JournalEntry(Journal):
    def __init__(self, date, session_log, audio_recording, resultsPlot):
        self.date = date
        self.session_log = session_log
        self.audio_recording = audio_recording
        self.resultsPlot = resultsPlot


class SaiBot:
    def __init__(self):
        # Load Classifier to class object
        self.saiBot = self.load_saiBot()

    def load_saiBot(self):
        # Attempt to load existing model. If model isn't found, create a new one
        sai_bot_path = 'res/classification_data/models/saibot.sav'
        try:
            print('Attempting to load SaiBot...')
            saiBot = joblib.load(sai_bot_path)
            print('Succesfully loaded SaiBot.')
            return saiBot
        except FileNotFoundError:
            print('saibot.sav not found. Setting up SaiBot...')
            print('Setting-Up SaiBot - Naive Bayes Classifier...')

            if not os.path.exists('res/classification_data/models'):
                os.makedirs('res/classification_data/models')

            # Setup NB Classification Pipeline
            saiBot = Pipeline([('vect', CountVectorizer()),
                               ('tfidf', TfidfTransformer()),
                               ('clf', MultinomialNB())])
            print("SaiBot - Features Extracted.")
            print("SaiBot - Term Frequencies Extracted.")
            print('SaiBot - Naive Bayes Classifier Setup Complete.')

            # Load datasets
            df = self.load_data()

            print(f'Head: {df.head()}')

            # Run Naive Bayes(NB) ML Algorithm to build model
            saiBot = saiBot.fit(df.prompt.values.astype('U'), df.response.values.astype('U'))

            # Test Performance of NB Classifier
            test_classifier(saiBot, df, 'SaiBot', 'prompt')

            # Save model
            joblib.dump(saiBot, sai_bot_path)

            return saiBot

    def load_data(self):
        # Configure filepath
        data_filepath = 'res/classification_data/datasets/SaiBotData.csv'

        # Try to load existing master dataset. If not found, return error.
        try:
            df = pd.read_csv(data_filepath)
            dfLength = len(df['prompt'])
            start_time = time.time()

            for i in range(0, dfLength):
                time_elapsed = (time.time() - start_time) / 60

                # Get the value of current selftext
                value = df['prompt'].iloc[i]

                # Clean the data
                print(value)
                value = text_cleanup(value)

                # Update the dataframe for master-set
                df['prompt'].iloc[i] = value
                prompt = df['prompt'].iloc[i]
                category = df['category'].iloc[i]

                # Progress Report
                print(f'Time Elapsed: {time_elapsed:.2f} | Category: {category} | {i}/{dfLength} | Prompt: {prompt}')

            return df
        except FileNotFoundError:
            print('SaiBotData.csv is missing. Please insert SaiBotData.csv into the '
                  '"res/classification_data/datasets" folder')

    def get_response(self, input_text):
        print(f'Input: {input_text}')
        cleaned_input = text_cleanup(input_text)
        print(f'Cleaned Input: {cleaned_input}')
        response = self.saiBot.predict([cleaned_input])
        print(f'Response: {response}')

        # Check if response is valid - Valid >= 30% chance
        proba = self.saiBot.predict_proba([input_text])
        return response[0]


class MentalHealthAnalyzer:
    def __init__(self, *args, **kwargs):
        self.text_clf = self.load_classifier()

    def load_classifier(self):
        # Attempt to load existing model. If model isn't found, create a new one
        nb_filename = 'res/classification_data/models/mha.sav'
        try:
            print('Attempting to load mha.sav...')
            text_clf = joblib.load(nb_filename)
            return text_clf
        except FileNotFoundError:
            print('mha.sav not found. Setting up NB Classification Model.')
            print('Setting-Up MHA - Naive Bayes Classifier...')

            if not os.path.exists('res/classification_data/models'):
                os.makedirs('res/classification_data/models')

            # Setup NB Classification Pipeline
            text_clf = Pipeline([('vect', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', MultinomialNB())])
            print("MHA - Features Extracted.")
            print("MHA - Term Frequencies Extracted.")
            print('MHA - Naive Bayes Classifier Setup Complete.')

            # Load datasets
            df = self.load_data()
            print(f'Head: {df.head()}')

            # Run Naive Bayes(NB) ML Algorithm to build model
            print('Fitting Model...')
            text_clf = text_clf.fit(df.selftext.values.astype('U'), df.category.values.astype('U'))
            print('Model has been fitted.\nSaving model to res/classification_data/models/mha.sav')

            # Test Performance of NB Classifier
            test_classifier(text_clf, df, 'MHA', 'selftext')

            # Save model
            joblib.dump(text_clf, nb_filename)
            print('Model saved.')

            return text_clf

    def load_data(self):
        # Configure Filepaths
        master_filepath = 'res/classification_data/datasets/master-set.csv'

        try:  # Try to load existing master dataset. If not found, create new one
            return pd.read_csv(master_filepath)
        except FileNotFoundError:
            # Removed r/suicidewatch as classifications were inaccurate due to lack of data/unable to collect more data
            filepath_dict = {'anxiety': 'res/classification_data/datasets/20k/anxiety.csv',
                             'depression': 'res/classification_data/datasets/20k/depression.csv',
                             'tourettes': 'res/classification_data/datasets/20k/tourettes.csv',
                             'adhd': 'res/classification_data/datasets/20k/adhd.csv',
                             'schizophrenia': 'res/classification_data/datasets/20k/schizophrenia.csv',
                             'eatingdisorder': 'res/classification_data/datasets/20k/eatingdisorder.csv',
                             'bipolar': 'res/classification_data/datasets/20k/bipolar.csv',
                             'ocd': 'res/classification_data/datasets/20k/ocd.csv'
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
            df = df.fillna('')

            dfLength = len(df['selftext'])

            start_time = time.time()
            for i in range(0, dfLength):
                time_elapsed = (time.time() - start_time) / 60

                # Get the value of current selftext
                value = df['selftext'].iloc[i]

                # Clean the data
                value = text_cleanup(value)

                # Update the dataframe for master-set
                df['selftext'].iloc[i] = value
                selftext = df['selftext'].iloc[i]
                category = df['category'].iloc[i]

                # Progress Report
                print(
                    f'Time Elapsed: {time_elapsed:.2f}m | Category: {category} | {i}/{dfLength} | Selftext: {selftext}')

            print(f'5 Samples: {df.head()}\n| Summary: \n{df.info}\nDescription: {df.describe()}\nShape: {df.shape}')

            # Make master-set csv and save .csv file
            df.to_csv('res/classification_data/datasets/master-set.csv', index=0)
            print('Master Dataset Created.')

            return df

    # Check the likelihood of different disorders
    def analyze_text(self, input_text):
        # Make a list of possible disorder/classes
        classes = self.text_clf.classes_.tolist()

        # Find/store % of chance that user is experiencing disorder(s)
        detailed_analysis = self.text_clf.predict_proba([input_text]).tolist()
        detailed_analysis = detailed_analysis[0]

        for i in range(0, len(detailed_analysis)):
            detailed_analysis[i] = (round(detailed_analysis[i] * 100, 2))

        print(f'Input Text: {input_text}')
        print(f'Classes: {classes}')
        print(f'Detailed Output: {detailed_analysis}')
        return detailed_analysis


class TTSThread(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True
        self.start()

    def run(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 185)
        engine.startLoop(False)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)  # voices[0] == male, voices[1] == female
        thread_running = True
        while thread_running:
            if self.queue.empty():
                engine.iterate()
            else:
                data = self.queue.get()
                if data == "exit":
                    thread_running = False
                else:
                    engine.say(data)
        engine.endLoop()


class STTThread:
    def __init__(self):
        # Setup STT Recognizer
        self.recognizer = sr.Recognizer()

        # Setup thread
        threading.Thread.__init__(self)
        self.listening = False
        self.stt_thread = threading.Thread(target=self.listen)
        self.stt_thread.daemon = True
        self.text = " "
        print('STT Thread Started.')

    def listen(self):
        print('Reached listen().')
        while self.listening:
            try:
                with sr.Microphone(0) as mic:
                    print('Listening...')
                    self.recognizer.adjust_for_ambient_noise(mic)  # Filter out background noise
                    audio = self.recognizer.record(source=mic, duration=5)  # Initialize input from mic

                    self.text = self.text + self.recognizer.recognize_google(audio)  # Convert audio to text
                    print(f'Voice Input: {self.text}')
            except Exception as e:
                print('Voice Input: None')

    def toggle(self):
        print('Reached toggle().')
        if self.listening:
            self.listening = False

            print('STT Thread Stopped.')
            return self.text
        else:
            self.listening = True
            try:
                self.stt_thread.start()
                print('STT Thread Started.')
            except:
                print('STT Thread is already running.')


class MainApp(tk.Tk):
    # init function for MainApp
    def __init__(self, *args, **kwargs):
        # init function for Tk
        tk.Tk.__init__(self, *args, **kwargs)

        # Create a container
        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        # initializes frames array
        self.frames = {}

        # Create class variables of ui images
        self.uiToggleImg = tk.PhotoImage(file="res/img/uiToggle.png").subsample(15, 15)
        self.homeBtnImg = tk.PhotoImage(file='res/img/home.png').subsample(15, 15)
        self.copingBtnImg = tk.PhotoImage(file='res/img/meditation.png').subsample(15, 15)
        self.journalBtnImg = tk.PhotoImage(file='res/img/journal.png').subsample(15, 15)

        # Toggle light/dark mode
        def uiMode():
            global darkUI
            if darkUI:
                print('Theme switched to light-mode.')
                sv_ttk.set_theme('light')
                darkUI = False
            else:
                print('Theme switched to dark-mode.')
                sv_ttk.set_theme('dark')
                darkUI = True

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (
                HomePage, TextSessionPage, VentingPage, CopingPage, ResultsPage, JournalPage, BreathingActivity,
                IdentifyingSurroundings):
            frame = F(container, self)

            # Setup window dimensions
            global window_width
            global window_height

            # light/dark mode toggle
            uiToggle = ttk.Button(self, image=self.uiToggleImg, width=10, command=lambda: uiMode())
            uiToggle.place(x=window_width - 70, y=70)

            # home button
            homeBtn = ttk.Button(self, image=self.homeBtnImg, width=10, command=lambda: self.show_frame(HomePage))
            homeBtn.place(x=window_width - 70, y=10)

            # coping exercises button
            # img src = "https://www.flaticon.com/free-icons/yoga" created Freepik - Flaticon
            copingBtn = ttk.Button(self, image=self.copingBtnImg, width=10, command=lambda: self.show_frame(CopingPage))
            copingBtn.place(x=window_width - 70, y=130)

            # img src = "https://www.flaticon.com/free-icons/journal" created by Freepik - Flaticon
            journalBtn = ttk.Button(self, image=self.journalBtnImg, width=10,
                                    command=lambda: self.show_frame(JournalPage))
            journalBtn.place(x=window_width - 70, y=190)

            # initializing frame of that object from each page planned
            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        # Show the home page
        self.show_frame(HomePage)

    # to display the current frame passed as parameter to switch to desired frame of program
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        frame.event_generate("<<ShowFrame>>")


class HomePage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)

        # Setup window dimensions
        global window_width
        global window_height

        # Body Frame
        body_frame = ttk.Frame(self, width=window_width, height=window_height - 400)
        body_frame.pack(side="top", pady=175, fill="x", expand=True)
        body_frame.anchor('center')

        # Buttons to choose session type
        # Text Session
        textSessionBtn = ttk.Button(body_frame, width=40, text="Start Text Session",
                                    command=lambda: {
                                        controller.show_frame(TextSessionPage),
                                        tts_queue.put(TextSessionPage.starter_text[5:])
                                    })
        textSessionBtn.pack(ipady=20, padx=10, pady=10)

        # Voice Session
        voiceSessionBtn = ttk.Button(body_frame, width=40, text="Start Vent Session",
                                     command=lambda: controller.show_frame(VentingPage))
        voiceSessionBtn.pack(ipady=20, padx=10, pady=10)

        # Coping/De-stressing Exercises Activities
        copingPageBtn = ttk.Button(body_frame, width=40, text="Coping Exercises",
                                   command=lambda: controller.show_frame(CopingPage))
        copingPageBtn.pack(ipady=20, padx=10, pady=10)

        # Footer Frame
        footer_frame = ttk.Frame(self, width=window_width, height=window_height - 200)
        footer_frame.pack(side="bottom", fill="x")

        # Disclaimer Label
        disclaimerLabel = ttk.Label(self,
                                    text='Disclaimer: This tool is not a replacement for professional psychiatric '
                                         'and/or therapeutric assistance.')
        disclaimerLabel.place(x=130, y=window_height - 30)


class TextSessionPage(ttk.Frame):
    # Class Fields
    lineCount = 0  # used to make sure new outputs go onto next line
    starter_text = 'Sai: Welcome to ESAI. My name is Sai and I am here to ' \
                   'provide you with emotional support as ' \
                   'needed. How are you feeling today?\n'

    user_input = ''

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)

        def jumpToResults():
            print('Reached jumpToResults().')
            # Get user input blob of text
            global mha_values
            global session_log
            mha_values['values'] = mha.analyze_text(self.user_input)

            # Jump to Results Page
            controller.show_frame(ResultsPage)

        # Setup window dimensions
        global window_width
        global window_height

        # Body Frame
        body_frame = ttk.Frame(self, width=window_width)
        body_frame.place(x=30, y=30)

        # Results button
        # https://www.flaticon.com/free-icons/notepad created by Freepik - Flaticon
        self.resultsBtnImg = tk.PhotoImage(file='res/img/results.png').subsample(15, 15)
        resultsBtn = ttk.Button(self, image=self.resultsBtnImg, width=10, command=jumpToResults)
        resultsBtn.place(x=window_width - 70, y=250)

        # Text Widget to Display Chat with Scrollbar
        self.output = tk.Text(body_frame, width=90, height=32)
        scrollBar = ttk.Scrollbar(body_frame, orient='vertical', command=self.output.yview)
        scrollBar.grid(column=1, row=0, sticky='nwes')
        self.output['yscrollcommand'] = scrollBar.set
        self.output.grid(column=0, row=0, sticky='ns')

        # Trigger initial output from Sai
        self.lineCount = 1.0
        self.output['state'] = 'normal'  # Re-enable editing to use insert()

        self.output.insert(self.lineCount, self.starter_text)
        self.output['state'] = 'disabled'  # Prevent user from editing output text

        # Footer Frame
        footer_frame = ttk.Frame(self, width=window_width)
        footer_frame.place(x=30, y=window_height - 70)

        # Entry Field
        self.input_field = ttk.Entry(footer_frame, width=77)
        self.input_field.pack(side='left', padx=5, pady=10)
        self.input_field.focus_set()  # Bring user to text field immediately

        # Setup binding
        self.input_field.bind('<Return>', self.setOutput)

        # Submit Button
        submitBtn = ttk.Button(footer_frame, text='Submit', width=8)
        submitBtn.pack(side='right', padx=5, pady=10)

        # Button is binded instead of command to prevent having to remake setOutput() function
        submitBtn.bind('<Button>', self.setOutput)

    def setOutput(self, bindArg):  # bindArg acts as a 2nd parameter to allow enter key to send input
        inputText = self.input_field.get()  # Get input text and store before erasing
        self.input_field.delete(0, 'end')  # Erase input field

        # Validate inputText is not null before continuing
        if len(inputText) >= 1:
            # Set User Output
            self.output['state'] = 'normal'  # Re-enable editing to use insert()
            self.lineCount = self.lineCount + 1
            self.output.insert(self.lineCount, ('You: ' + inputText + "\n"))
            self.output['state'] = 'disabled'  # Prevent user from editing output text
            self.user_input = self.user_input + inputText + '. '  # Append the user_input string

            # Append session logs
            global session_log
            session_log['speaker'].append('You')
            session_log['dialogue'].append(inputText)

            # Set Sai's Output
            self.output['state'] = 'normal'  # Re-enable editing to use insert()
            self.lineCount = self.lineCount + 1
            response = self.getResponse(inputText)
            self.output.insert(self.lineCount, response + "\n")
            self.output['state'] = 'disabled'  # Prevent user from editing output text
            tts_queue.put(response[5:])  # the :5 is to prevent tts from including sai's name

            # Append session logs
            session_log['speaker'].append('Sai')
            session_log['dialogue'].append(response[5:])
            print(f'\nSession Log: {session_log.items()}')

    # Get response based on the users input and return it to be printed under Sai's response
    def getResponse(self, inputText):
        # Check to see if any flags are triggered (chance of disorder > 50%)
        disorders = mha.text_clf.classes_.tolist()

        report = mha.analyze_text(self.user_input)

        print(f'\nDetailed Analysis: {report}')

        for i in range(0, len(report)):
            print(report[i])
            if report[i] > 50:  # If there is a 50% or higher chance of illness, add it to detected illness list
                print(f'\n{disorders[i]} Detected.')
            else:
                print(f'\n{disorders[i]}: Not Detected.')

        # Get Sai's response
        inputText = text_cleanup(inputText)
        response = sai_bot.get_response(inputText)
        print(f'Response: {response}')
        if response is not None:
            return 'Sai: ' + response
        else:
            return "Sai: I'm sorry, I do not understand."


class VentingPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)

        # Variable to track if recording is in progress
        self.recording = False
        self.pyAudio = pyaudio.PyAudio()

        self.frames = []  # arr to store frames
        # Configure stream
        self.chunk = 1024  # Record in chunks of 1024 samples
        self.sample_format = pyaudio.paInt16  # 16 bits per sample
        self.channels = 1
        self.fs = 44100  # Record at 44100 samples per second

        self.stream = self.pyAudio.open(format=self.sample_format,
                                        channels=self.channels,
                                        rate=self.fs,
                                        frames_per_buffer=self.chunk,
                                        input=True)

        # Body Frame
        global window_width
        global window_height

        body_frame = ttk.Frame(self, width=window_width, height=window_height - 400)
        body_frame.pack(side="top", pady=75, fill="x", expand=True)
        body_frame.anchor('center')

        # Configure status label
        self.status_label = ttk.Label(body_frame, text='Ready')
        self.status_label.pack(padx=10, pady=10)

        # Configure recording button
        self.recordingBtn = ttk.Button(body_frame, text='Start Recording', command=self.start_recording)
        self.recordingBtn.pack(padx=10, pady=10)

        # Footer Frame
        footer_frame = ttk.Frame(self, width=window_width, height=window_height - 200)
        footer_frame.pack(side="bottom", fill="x")

    def start_recording(self):
        print('Reached start_recording()')
        # Update button and status
        self.recordingBtn.config(text='Stop Recording', command=self.stop_recording)
        self.status_label.config(text='Recording...')
        self.status_label.update()

        # Start the recording
        self.recording = True
        recording_thread = threading.Thread(target=self.record)
        recording_thread.start()

    def stop_recording(self):
        print('Reached stop_recording()')
        # Update button
        self.recordingBtn.config(text='Start Recording', command=self.start_recording)

        # Stop Recording
        self.recording = False
        # self.stream.stop_stream()
        # self.stream.close()

        # Terminate pyaudio interface
        # self.pyAudio.terminate()

        print('Finished Recording')

        # Generate unique filepath
        now = datetime.datetime.now()
        currentDateTime = now.strftime("%m_%d_%Y-%H_%M_%S")
        filepath = r'./UserData/audio_recordings/%s.wav' % currentDateTime

        # Save the recording as WAV file
        wave_file = wave.open(filepath, 'wb')
        wave_file.setnchannels(self.channels)
        wave_file.setsampwidth(self.pyAudio.get_sample_size(self.sample_format))
        wave_file.setframerate(self.fs)
        wave_file.writeframes(b''.join(self.frames))
        wave_file.close()

        # Update status
        status = f'Saved recording to: {filepath}'
        self.status_label.config(text=status)
        self.status_label.update()

    def record(self):
        # Initialize array to store frames as they come
        while self.recording:
            print('Recording...')
            data = self.stream.read(self.chunk)
            self.frames.append(data)


class CopingPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        # Setup window dimensions
        global window_width
        global window_height

        # Body Frame
        body_frame = ttk.Frame(self, width=window_width, height=window_height - 400)
        body_frame.pack(side="top", pady=75, fill="x", expand=True)
        body_frame.anchor('center')

        # Buttons to choose session type
        # Page title label
        copingPageLbl = ttk.Label(body_frame, text='Coping Activites')
        copingPageLbl.pack(padx=10, pady=10)

        # Breathing Activity
        breathingBtn = ttk.Button(body_frame, text='Breathing Activity',
                                  command=lambda: controller.show_frame(BreathingActivity))
        breathingBtn.pack(ipady=20, ipadx=20, padx=10, pady=10)

        # Identifying Surrounds Activitiy
        surroundingsBtn = ttk.Button(body_frame, text='Identifying Surroundings Activity',
                                     command=lambda: controller.show_frame(IdentifyingSurroundings))
        surroundingsBtn.pack(ipady=20, ipadx=20, padx=10, pady=10)

        # Footer Frame
        footer_frame = ttk.Frame(self, width=window_width, height=window_height - 200)
        footer_frame.pack(side="bottom", fill="x")


class BreathingActivity(ttk.Frame):
    def __init__(self, parent, controller):
        print('Reached BreathingActivity.')
        ttk.Frame.__init__(self, parent)

        # Setup window dimensions
        global window_width
        global window_height

        # Body Frame
        body_frame = ttk.Frame(self, width=window_width, height=window_height - 400)
        body_frame.pack(side="top", pady=75, fill="x", expand=True)
        body_frame.anchor('center')

        # Setup variables
        self.instruction = 'Please press the "Start" button and close your eyes to begin the breathing activity.'

        # Label to store instructions
        self.instruction_label = ttk.Label(body_frame, text=self.instruction)
        self.instruction_label.pack(padx=10, pady=10)

        # Setup the thread for the activity to prevent tkinter from freezing
        self.breathing_thread = threading.Thread(target=self.start_breathing)

        # Button to start activity thread
        start_button = ttk.Button(body_frame, text="Start", command=self.start_thread)
        start_button.pack(padx=10, pady=10)

        # Footer Frame
        footer_frame = ttk.Frame(self, width=window_width, height=window_height - 200)
        footer_frame.pack(side="bottom", fill="x")

        # Bind the activity trigger to when frame is visible
        self.bind('<<ShowFrame>>', self.start_activity)

    def start_thread(self):
        try:
            print('Started Breathing Thread.')
            self.breathing_thread.start()  # Will start if thread isn't running.
        except RuntimeError:
            print('Breathing Thread is already running.')
            self.start_breathing()

    def start_breathing(self):
        def breathe_in():
            self.instruction_label.config(text='Breathe in...')
            print('Breathing in...')
            tts_queue.put('Breathe in.')

        def breathe_out():
            self.instruction_label.config(text='Breathe out...')
            print('Breathing out...')
            tts_queue.put('Breathe out.')

        def hold_breathe():
            self.instruction_label.config(text='Hold...')
            print('Holding...')
            tts_queue.put('Hold.')

        #  Breathing activity
        for i in range(5):  # 5 Rounds at 4 seconds each
            print(f'Breathing Round {i + 1}/5')
            self.after(4000, breathe_in())
            self.instruction_label.update()

            self.after(4000, hold_breathe())
            self.instruction_label.update()

            self.after(4000, breathe_out())
            self.instruction_label.update()

            self.after(4000, hold_breathe())
            self.instruction_label.update()

        # End activity
        activity_status = 'Breathing activity completed'
        print(activity_status)
        self.instruction_label.config(text=activity_status)
        tts_queue.put('Great job! You have finished the breathing activity. '
                      'I hope this helped you de-stress at least a little bit.')

    def start_activity(self, bindArgs):
        # Send instruction to tts_queue for speech
        tts_queue.put(self.instruction)


class IdentifyingSurroundings(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)

        print('Reached IdentifyingSurroundings.')

        # Setup window dimensions
        global window_width
        global window_height

        # Body Frame
        body_frame = ttk.Frame(self, width=window_width, height=window_height - 400)
        body_frame.pack(side="top", pady=75, fill="x", expand=True)
        body_frame.anchor('center')

        # Setup variables
        self.instruction = 'Please press the "Start" button to begin the Identifying Surroundings activity.'

        # Label to store instructions
        self.instruction_label = ttk.Label(body_frame, text=self.instruction)
        self.instruction_label.pack(padx=10, pady=10)

        # Setup the thread for the activity to prevent tkinter from freezing
        self.identifying_thread = threading.Thread(target=self.start_identifying)

        # Button to start activity thread
        start_button = ttk.Button(body_frame, text="Start", command=self.start_thread)
        start_button.pack(padx=10, pady=10)

        # Footer Frame
        footer_frame = ttk.Frame(self, width=window_width, height=window_height - 200)
        footer_frame.pack(side="bottom", fill="x")

        # Bind the activity trigger to when frame is visible
        self.bind('<<ShowFrame>>', self.start_activity)

    def start_activity(self, bindArgs):
        tts_queue.put(self.instruction)

    def start_identifying(self):
        print('Reached start_identifying().')

        # Start activity

    def start_thread(self):
        try:
            print('Started Identifying Thread.')
            self.identifying_thread.start()  # Will start if thread isn't running.
        except RuntimeError:
            print('Identifying Thread is already running.')
            self.start_identifying()


class JournalPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        print('Reached JournalPage.')

        # images
        # "https://www.flaticon.com/free-icons/back" created by Roundicons - Flaticon
        self.leftArrowImg = tk.PhotoImage(file="res/img/left-arrow.png").subsample(15, 15)

        # <a href="https://www.flaticon.com/free-icons/next" created by Roundicons - Flaticon
        self.rightArrowImg = tk.PhotoImage(file="res/img/right-arrow.png").subsample(15, 15)

        # Get current date
        self.today_date = datetime.datetime.today()


        # Setup window dimensions
        global window_width
        global window_height

        # Header Frame
        header_frame = ttk.Frame(self, width=window_width, height=70)
        header_frame.pack(side="top", fill='x')

        # Date Selection Frame
        date_selection_frame = ttk.Frame(header_frame, width=200, height=window_height)
        # date_selection_frame.place(x=window_width-310, y=0)
        date_selection_frame.pack()

        # Date Label and Arrow Buttons
        left_arrow = ttk.Button(date_selection_frame, image=self.leftArrowImg, command=self.move_date_back)
        left_arrow.pack(side='left', padx=10, pady=10)

        date_label = ttk.Label(date_selection_frame, text='xx/xx/xxxx')
        date_label.pack(side='left', padx=10, pady=10)

        left_arrow = ttk.Button(date_selection_frame, image=self.rightArrowImg, command=self.move_date_back)
        left_arrow.pack(side='right', padx=10, pady=10)

        # Session Logs and recordings Frame
        logs_frame = tk.Frame(self)
        logs_frame.place(x=30, y=80)

        # Tabbed widget for Logs and Audio
        tabControl = ttk.Notebook(logs_frame)
        logs_tab = tk.Frame(tabControl, width=380, height=480, bg='grey')
        recordings_tab = tk.Frame(tabControl, width=380, height=480, bg='grey')

        # Add tabs
        tabControl.add(logs_tab, text='Session Logs')
        tabControl.add(recordings_tab, text='Audio Recordings')

        # Finalize widget
        tabControl.pack(expand=1, fill='both')

        # Results frame
        results_frame = tk.Frame(self)
        results_frame.place(x=window_width-440, y=80)

        # Tabbed widget for Logs and Audio
        result_tabControl = ttk.Notebook(results_frame)
        today_results_tab = tk.Frame(result_tabControl, width=340, height=480)
        month_results_tab = tk.Frame(result_tabControl, width=340, height=480)
        year_results_tab = tk.Frame(result_tabControl, width=340, height=480)
        all_results_tab = tk.Frame(result_tabControl, width=340, height=480)

        # Add tabs
        result_tabControl.add(today_results_tab, text='Today')
        result_tabControl.add(month_results_tab, text='Month')
        result_tabControl.add(year_results_tab, text='Year')
        result_tabControl.add(all_results_tab, text='All Time')

        # Finalize widget
        result_tabControl.pack(expand=1, fill='both')

        # Bind the activity trigger to when frame is visible
        self.bind('<<ShowFrame>>', self.update_journal(self.today_date))

    def move_date_back(self):
        print('Reached move_date_back()')

    def move_date_forward(self):
        print('Reached move_date_forward()')

    def update_journal(self, today):
        print("Reached pull_recent_journal()")

class ResultsPage(ttk.Frame):
    def __init__(self, parent, mha_values):
        ttk.Frame.__init__(self, parent)
        print('Reached Results Page')

        def show_results(bindArg):
            global mha_values
            print('Reached show_results.')
            print(mha_values['values'][0])

            # Replot data with new values
            axes.bar(mha_values['categories'], mha_values['values'], width=0.7)

            figure_canvas.draw()

        # Setup window dimensions
        global window_width
        global window_height

        # Body Frame
        body_frame = ttk.Frame(self, width=window_width, height=window_height - 400)
        body_frame.pack(side="top", pady=(100, 75), fill="x", expand=True)
        body_frame.anchor('center')

        # Make bar chart
        figure = plt.Figure(figsize=(7, 3), dpi=100)
        figure_canvas = FigureCanvasTkAgg(figure, body_frame)
        axes = figure.add_subplot()
        axes.set_title('Mental Health Analysis')
        # plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")  # Put labels at an angle

        figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.Y, expand=1)

        # Results Page Header Label
        resultsLabel = ttk.Label(self, text='Session Evaluation')
        resultsLabel.place(x=window_width / 2 - 50, y=50)

        buttonBar = ttk.Frame(body_frame)
        buttonBar.pack(padx=10, pady=(40, 0))

        # Save session logs
        saveLogBtn = ttk.Button(buttonBar, text='Save Session Logs', command=self.export_session_log)
        saveLogBtn.grid(row=0, column=0, padx=5, pady=5)

        # Save recording
        saveRecordingBtn = ttk.Button(buttonBar, text='Save Session Recording', command=self.export_vent)
        saveRecordingBtn.grid(row=0, column=1, padx=5, pady=5)

        # Disclaimer Label
        disclaimerLabel = ttk.Label(self,
                                    text='Disclaimer: This tool is not a replacement for professional psychiatric '
                                         'and/or therapeutric assistance.')
        disclaimerLabel.pack()

        # Footer Frame
        footer_frame = ttk.Frame(self, width=window_width, height=window_height - 200)
        footer_frame.pack(side="bottom", fill="x")

        # Bind showframe event to updating the results
        self.bind('<<ShowFrame>>', show_results)

    def export_session_log(self):
        print('Exporting session logs...')

        # Convert session log to dataframe
        session_df = pd.DataFrame(session_log)

        # Initialize word doc
        doc = docx.Document()

        # Initialize Table
        output_table = doc.add_table(rows=session_df.shape[0], cols=session_df.shape[1])

        # Add session log data to the table
        for i in range(session_df.shape[0]):
            for j in range(session_df.shape[1]):
                cell = session_df.iat[i, j]
                output_table.cell(i, j).text = str(cell)

        directory = filedialog.askdirectory(title='Select a File')
        # Save the word doc
        doc.save(f'{directory}/ESAI-Session_Log.docx')

        print('Exported session logs.')

    def export_vent(self):
        print('Exporting vent recording...')

        # Code

        print('Exported vent recording.')


# Start the program
if __name__ == "__main__":
    print('Launching ESAI...')

    # Setup TTS
    tts_queue = queue.Queue()
    print('TTS Queue Created.')

    tts_thread = TTSThread(tts_queue)
    print('TTS Thread Started.')

    # Setup STT
    stt = STTThread()

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
    all_stopwords.remove('you')
    all_stopwords.remove('your')
    all_stopwords.remove('name')
    all_stopwords.remove('who')
    all_stopwords.remove('me')
    all_stopwords.add("/")
    all_stopwords.add('.')
    all_stopwords.add(",")
    all_stopwords.add("'")

    print('Spacy NLP has Loaded Successfully.')
    print(f'Stopwords: {all_stopwords}')

    # Setup Chat Bot
    sai_bot = SaiBot()

    # Global scope vars and structs
    mha_values = {
        'categories': ['ADHD', 'Anxiety', 'Bipolar', 'Depression',
                       'ED', 'OCD', 'Schizo.', "Tourette's"],
        'values': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }

    session_log = {
        'speaker': ['Speaker'],
        'dialogue': ['Dialogue']
    }

    # Make directories if they don't exist
    try:
        recording_path = os.path.join(os.curdir, 'UserData/audio_recordings')
        log_path = os.path.join(os.curdir, 'UserData/session_logs')
        journal_path = os.path.join(os.curdir, 'UserData/journals')

        # Audio Recordings
        if not os.path.exists(recording_path):
            os.makedirs(recording_path)
            print('Successfully created audio_recording directory.')

        # Session Logs
        if not os.path.exists(log_path):
            os.makedirs(log_path)
            print('Successfully created session_logs directory.')

        # Journals
        if not os.path.exists(journal_path):
            os.makedirs(journal_path)
            print('Successfully created journals directory.')
    except OSError as error:
        print(f'Unable to create required directories. Error: {error}')

    # Load / Create Journal for the month
    journal = Journal()
    journal.loadJournal()

    # Setup window dimensions
    window_width = 870
    window_height = 640

    # Setup MainApp
    darkUI = True
    app = MainApp()
    app.title("ESAI: An Emotional Support AI")

    # Setup tkinter theme
    sv_ttk.set_theme("dark")

    # Load MentalHealthAnalyzer / Naive Bayes Classifier
    mha = MentalHealthAnalyzer()

    # Get screen dimensions
    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()

    # Find the center point
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)

    # Configure Window
    app.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    app.resizable(width=False, height=False)  # Prevent Resizing
    app.rowconfigure(3)
    app.columnconfigure(3)

    print('ESAI Has Started Successfully.')
    app.mainloop()  # Start app and keep running
    print('ESAI terminated.')
