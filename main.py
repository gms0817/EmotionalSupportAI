# Imports
import datetime
import os
import pickle
import re
import threading
import time
import tkinter as tk
import wave
from datetime import timedelta
from tkinter import *
from tkinter import ttk, filedialog
import contractions
import docx  # Install python-docx not docx for python 3.9
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyaudio
import pyttsx3
import spacy
import speech_recognition as sr
import sv_ttk
import unidecode
import openai
from tkcalendar import Calendar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class Journal:
    def __init__(self):
        # List of Journal Entries
        self.entryList = [JournalEntry() for i in range(31)]

        # Monthly Average of MHA_values
        self.month_avg_mha_values = {
            'categories': ['ADHD', 'Anxiety', 'Bipolar', 'Depression',
                           'ED', 'OCD', 'Schizo.', "Tourette's"],
            'values': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }

        # Annual Average of MHA_values
        self.annual_avg_mha_values = {
            'categories': ['ADHD', 'Anxiety', 'Bipolar', 'Depression',
                           'ED', 'OCD', 'Schizo.', "Tourette's"],
            'values': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }

    # Add journal entry
    def addEntry(self, journalEntry):
        print('Reached addEntry()')
        self.entryList.append(journalEntry)
        self.exportJournal()  # Update journal

    # Update mha_valus avgs
    def update_monthly_mha_avgs(self):
        print('Reached update_mha_avgs()')

        # Reset the average before calculating
        self.month_avg_mha_values['values'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Update MHA Monthly Average
        # iterate through Journal Entries
        entry_count = 0
        for i in range(len(self.entryList)):
            entry = self.entryList[i]
            entry_count = entry_count + 1
            for j in range(len(entry.mha_values['values'])):
                if entry.mha_values['values'][j] != 0.0:  # Check for invalid entryt
                    self.month_avg_mha_values['values'][j] = (self.month_avg_mha_values['values'][j] +
                                                              entry.mha_values['values'][j]) / entry_count
                else:
                    entry_count = entry_count - 1
                    break
        print(f'Number of Valid Entries: {entry_count}')
        print(f'Updated Month MHA Values: {self.month_avg_mha_values}')

    def update_annual_mha_avgs(self):
        # Update MHA Annual Average
        # Find current year
        current_year = datetime.datetime.now().strftime('%Y')

        # Iterate through journals
        entry_count = 0

        for current_journal in os.listdir(journal_path):
            if current_year in current_journal:
                # Load current journal
                file_handler = open(path, 'rb')
                temp_journal = pickle.load(file_handler)
                file_handler.close()
                annualEntryList = temp_journal.entryList

                # iterate through Journal Entries
                for i in range(len(annualEntryList)):
                    entry = annualEntryList[i]
                    entry_count = entry_count + 1
                    for j in range(len(entry.mha_values['values'])):
                        if entry.mha_values['values'][j] != 0.0:  # Check for invalid entry
                            self.annual_avg_mha_values['values'][j] = (self.annual_avg_mha_values['values'][j] +
                                                                       entry.mha_values['values'][j]) / entry_count
                        else:
                            entry_count = entry_count - 1
                            break
        print(f'Number of Valid Entries: {entry_count}')
        print(f'Updated Annual MHA Values: {self.month_avg_mha_values}')

        # Update MHA Yearly Average

    # Export journal file
    def exportJournal(self):
        print('Reached exportJournal()')
        # Dump the file
        file = open(path, 'wb')
        pickle.dump(self, file)


class JournalEntry(Journal):
    def __init__(self):
        self.date = datetime.datetime.today().strftime('%B %d, %Y')
        self.session_log = {
            'dateTime': ['Date/Time'],
            'speaker': ['Speaker'],
            'dialogue': ['Dialogue']
        }
        # MHA Values Per Day
        self.mha_values = {
            'categories': ['ADHD', 'Anxiety', 'Bipolar', 'Depression',
                           'ED', 'OCD', 'Schizo.', "Tourette's"],
            'values': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }

    # Load Entry from Journal
    def loadEntry(self, entryDate):
        try:
            self.session_log = journal.entryList[entryDate - 1].session_log
            self.mha_values = journal.entryList[entryDate - 1].mha_values
            print(f'Successfully loaded existing entry.')
            print(f'Loaded Session Log: {self.session_log}')
            print(f'Loaded MHA Results: {self.mha_values}')


        except:
            print(f'No entry found for today. Default values assigned.')
            journal.addEntry(self)
            print('Entry added to Journal.')

    def export_session_log(self):
        print('Exporting session logs...')

        # Convert session log to dataframe
        session_df = pd.DataFrame(journalEntry.session_log)

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


class SaiBot:
    def __init__(self):
        # Load Classifier to class object
        openai.api_key = self.get_api_key()

    def get_api_key(self):
        print('Reached get_api_key()')
        try:
            print('Attempting to load OpenAI API Key...')
            with open('res/api_keys/open_ai.txt') as f:
                api_key = f.read()
            return api_key

        except FileNotFoundError:
            print('Unable to load OpenAI API Key')

    def get_response(self, input_text):
        print(f'Input: {input_text}')

        # Get response from ChatGPT API
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                # System message - Change name, role, and set the rules for messages.
                {"role": "system", "content": 'Refer to yourself as Sai.'
                                              'You are an Emotional Support AI powered by ChatGPT-turbo-3.5. '
                                              'When answering the user, respond as if you were their therapist.'
                                              'Although you will respond like a therapist, you are not a replacement '
                                              'for therapy and professional psychiatric help.'
                                              'Keep your responses to one paragraph maximum.'},

                # User message
                {"role": "user", "content": input_text}

            ]
        )
        return response['choices'][0]['message']['content']


class MentalHealthAnalyzer:
    def __init__(self, *args, **kwargs):
        self.text_clf = self.load_classifier()

    def text_cleanup(self, text):
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
            self.test_classifier(text_clf, df)

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
                value = self.text_cleanup(value)

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

    def plot_training_results(self, pass_score_dict, fail_score_dict):
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
        plt.title(f'MHA Accuracy')
        plt.xlabel("Number of Data Samples")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        plt.show()  # Show the scatter plot

    def test_classifier(self, model, df):
        print('Reached test_classifier()')
        dataName = 'MHA'
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
            actual = shuffled_df['category'].values[i]
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
            test_result = f'Classifier: MHA | ' \
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
        test_df.to_csv(f'res/classification_data/datasets/mha-test_results.csv', index=0)
        print("Detailed Testing Complete - test_results.csv created.")

        # General Performance Measuring of NB Classifier
        predicted = model.predict(df[dataName])
        score = np.mean(predicted == df['category'])
        time_elapsed = (time.time() - start_time) / 60

        print(f'Performance Analysis Completed in {time_elapsed:.2f} minutes.')
        print(f'Average Performance (Naive Bayes): {score:.2f}%')

        # Plot the performance of the NB Classifier
        self.plot_training_results(pass_score_dict, fail_score_dict)

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


# Text to Speech
class TTS:
    def __init__(self):
        # Pyttsx3 - No Internet
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 185)
        self.voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', self.voices[1].id)  # voices[0] == male, voices[1] == female

    def speak(self, data):
        def run():
            try:  # Attempt to speak
                self.engine.say(data)
                self.engine.runAndWait()
            except Exception as e:
                print('Unable to speak using pyttsx3. Please check installation.')
                print(e)

        # Create thread to run speech
        speech_thread = threading.Thread(target=run)
        speech_thread.start()


# Speech to Text
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
                HomePage, TextSessionPage, VentingPage, CopingPage, JournalPage, BreathingActivity):
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
                                    command=lambda: controller.show_frame(TextSessionPage))
        textSessionBtn.pack(ipady=20, padx=10, pady=10)

        # Voice Session
        VentingBtn = ttk.Button(body_frame, width=40, text="Start Vent Session",
                                command=lambda: controller.show_frame(VentingPage))
        VentingBtn.pack(ipady=20, padx=10, pady=10)

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
    starter_text = 'Sai: Welcome to your Emotional Support AI Experience. My name is Sai and I am here to ' \
                   'provide you with emotional support as ' \
                   'needed. How are you feeling today?\n'

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        # Set count of visits
        self.visits = 0

        def jumpToJournal():
            print('Reached jumpToJournal().')
            # Jump to Journal Page
            controller.show_frame(JournalPage)

        # Setup window dimensions
        global window_width
        global window_height

        # Body Frame
        body_frame = ttk.Frame(self, width=window_width)
        body_frame.place(x=30, y=30)

        # Text Widget to Display Chat with Scrollbar
        self.output = tk.Text(body_frame, width=80, height=28)
        self.output.configure(font=public_font)
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

        # Submit Button
        submitBtn = ttk.Button(footer_frame, text='Submit', width=8)
        submitBtn.pack(side='right', padx=5, pady=10)

        # Setup binding
        self.input_field.bind('<Return>', self.setOutput)
        self.bind('<<ShowFrame>>', self.load_page)
        submitBtn.bind('<Button>', self.setOutput)

    def load_page(self, bindArgs):
        print('Reached load_page()')
        if self.visits == 0:
            print('TextSessionPage: starter_text spoken.')
            tts.speak(self.starter_text[5:])
            self.visits = self.visits + 1
            self.setOutput(bindArgs)
        else:
            self.setOutput(bindArgs)

    def setOutput(self, bindArgs):  # bindArg acts as a 2nd parameter to allow enter key to send input
        inputText = self.input_field.get()  # Get input text and store before erasing
        self.input_field.delete(0, 'end')  # Erase input field

        # Grab global user input value
        global user_input

        # Get current datetime
        now = datetime.datetime.now()
        currentDateTime = now.strftime("%H:%M:%S")

        # Validate inputText is not null before continuing
        if len(inputText) >= 1:
            # Set User Output
            self.output['state'] = 'normal'  # Re-enable editing to use insert()

            self.lineCount = self.lineCount + 1
            self.output.insert(self.lineCount, ('You: ' + inputText + "\n"))
            self.output['state'] = 'disabled'  # Prevent user from editing output text
            user_input = user_input + inputText + '. '  # Append the user_input string

            # Append session logs
            journalEntry.session_log['dateTime'].append(currentDateTime)
            journalEntry.session_log['speaker'].append('You')
            journalEntry.session_log['dialogue'].append(inputText)

            # Set Sai's Output
            self.output['state'] = 'normal'  # Re-enable editing to use insert()
            self.lineCount = self.lineCount + 1
            self.output.insert(self.lineCount, '\n\n')  # Add space
            self.lineCount = self.lineCount + 1
            response = self.getResponse(inputText)
            self.output.insert(self.lineCount, response + "\n")
            self.lineCount = self.lineCount + 1
            self.output.insert(self.lineCount, '\n\n')  # Add space
            self.output['state'] = 'disabled'  # Prevent user from editing output text
            tts.speak(response[5:])

            # Append session logs
            journalEntry.session_log['dateTime'].append(currentDateTime)
            journalEntry.session_log['speaker'].append('Sai')
            journalEntry.session_log['dialogue'].append(response[5:])
            print(f'\nSession Log: {journalEntry.session_log.items()}')

            # Update MHA Values
            journalEntry.mha_values['values'] = mha.analyze_text(user_input)
            journal.exportJournal()  # Export changes to journal from session_logs and mha_values

    # Get response based on the users input and return it to be printed under Sai's response
    def getResponse(self, inputText):
        # Check to see if any flags are triggered (chance of disorder > 50%)
        disorders = mha.text_clf.classes_.tolist()

        report = mha.analyze_text(user_input)

        print(f'\nDetailed Analysis: {report}')

        for i in range(0, len(report)):
            print(report[i])
            if report[i] > 50:  # If there is a 50% or higher chance of illness, add it to detected illness list
                print(f'\n{disorders[i]} Detected.')
            else:
                print(f'\n{disorders[i]}: Not Detected.')

        # Get Sai's response
        # inputText = text_cleanup(inputText) No longer needed when using chatgpt api for text processing
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
        filepath = './UserData/audio_recordings/%s.wav' % currentDateTime

        # Update status
        status = f'Saving Recording to "{filepath}"...'
        self.status_label.config(text=status)
        self.status_label.update()

        # Save the recording as WAV file
        wave_file = wave.open(filepath, 'wb')
        wave_file.setnchannels(self.channels)
        wave_file.setsampwidth(self.pyAudio.get_sample_size(self.sample_format))
        wave_file.setframerate(self.fs)
        wave_file.writeframes(b''.join(self.frames))
        wave_file.close()

        # Update status
        status = f'Saved recording to: {filepath}.\nTranscribing audio...'
        self.status_label.config(text=status)
        self.status_label.update()

        # Transcribe audio to text
        with sr.AudioFile(filepath) as source:
            # Adjust recording for ambience
            stt.recognizer.adjust_for_ambient_noise(source)

            # Feed recording into recorder
            audio_data = stt.recognizer.record(source)

            # Transcribe the audio
            try:
                global user_input
                transcribed_text = stt.recognizer.recognize_google(audio_data)
                user_input = user_input + transcribed_text + '.'
            except Exception as e:
                print('No text transcribed. Audio file may not contain spoken words.')

        # Update mha_values based on new user_input values
        journalEntry.mha_values['values'] = mha.analyze_text(user_input)

        # Update session logs
        journalEntry.session_log['dateTime'].append(datetime.datetime.now().strftime('%H:%M:%S'))
        journalEntry.session_log['speaker'].append('You(Audio Transcription)')
        journalEntry.session_log['dialogue'].append(transcribed_text + ".")

        # Save updates/changes to Journal
        journal.exportJournal()

        print(f'Recording Transcript: {transcribed_text}')
        print(f'New user_input: {user_input}')

        # Update status
        status = f'Audio Recording Transcribed. Venting Session Completed.'
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
            tts.speak('Breathe in...')

        def breathe_out():
            self.instruction_label.config(text='Breathe out...')
            print('Breathing out...')
            tts.speak('Breathe out...')

        def hold_breathe():
            self.instruction_label.config(text='Hold...')
            print('Holding...')
            tts.speak('Hold...')

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
        tts.speak('Great job! You have finished the breathing activity. '
                  'I hope this helped you de-stress at least a little bit.')

    def start_activity(self, bindArgs):
        # Send instruction to tts for speech
        tts.speak(self.instruction)


class JournalPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        print('Reached JournalPage.')

        # Set linecount to track and increment as data populates
        self.lineCount = 1.0

        # images
        # "https://www.flaticon.com/free-icons/back" created by Roundicons - Flaticon
        self.leftArrowImg = tk.PhotoImage(file="res/img/left-arrow.png").subsample(15, 15)

        # "https://www.flaticon.com/free-icons/next" created by Roundicons - Flaticon
        self.rightArrowImg = tk.PhotoImage(file="res/img/right-arrow.png").subsample(15, 15)

        # "https://www.flaticon.com/free-icons/calendar" created by Freepik - Flaticon
        self.calendarImg = tk.PhotoImage(file="res/img/calendar.png").subsample(15, 15)

        # Get current date
        self.journalDate = journalEntry.date
        self.date = datetime.datetime.strptime(self.journalDate, '%B %d, %Y')

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

        self.date_label = ttk.Label(date_selection_frame, text=self.journalDate)
        self.date_label.pack(side='left', padx=10, pady=10)

        right_arrow = ttk.Button(date_selection_frame, image=self.rightArrowImg, command=self.move_date_forward)
        right_arrow.pack(side='right', padx=10, pady=10)

        # Calendar / Date-selection button
        calendarBtn = ttk.Button(header_frame, image=self.calendarImg, command=self.open_calendar_popup)
        calendarBtn.place(x=10, y=10)

        # Session Logs and recordings Frame
        logs_frame = ttk.Frame(self)
        logs_frame.place(x=30, y=80)

        # Tabbed widget for Logs and Audio
        tabControl = ttk.Notebook(logs_frame)

        # Session Logs
        logs_tab = tk.Frame(tabControl, width=380, height=480)

        # Scrollbar
        ver_scrollbar = ttk.Scrollbar(logs_tab, orient='vertical')
        ver_scrollbar.pack(side=RIGHT, fill='y')

        # Display logs
        self.logs_text = tk.Text(logs_tab, width=42, height=26.2, yscrollcommand=ver_scrollbar.set)
        self.logs_text.configure(font=public_font)
        self.logs_text['state'] = 'disabled'  # Prevent additional changes
        ver_scrollbar.config(command=self.logs_text.yview)
        self.logs_text.pack()

        # Audio Recordings
        recordings_tab = tk.Frame(tabControl, width=380, height=480)
        self.audio_file_list = os.listdir(recording_path)  # Get list of audio files

        # Setup ListBox to show files in recordings folder
        self.recordings_box = tk.Listbox(recordings_tab)
        self.recordings_box.pack(side=LEFT, expand=1, fill=BOTH)

        # Populate the recordings box
        for item in self.audio_file_list:
            self.recordings_box.delete(0, END)
            self.recordings_box.insert(END, item)

        # Play Recording Btn
        self.playBtn = ttk.Button(recordings_tab, text='Play Recording', command=self.play_recording)
        self.playBtn.place(x=120, y=440)

        # Add tabs for selection
        tabControl.add(logs_tab, text='Session Logs')
        tabControl.add(recordings_tab, text='Audio Recordings')

        # Finalize widget
        tabControl.pack(expand=1, fill='both')

        # Results frame
        results_frame = tk.Frame(self)
        results_frame.place(x=window_width - 420, y=80)

        # Tabbed widget for Logs and Audio
        result_tabControl = ttk.Notebook(results_frame)

        # Default values for charts
        chart_width = 3.2
        chart_height = 4.8

        # Today MHA Results
        self.today_results_tab = tk.Frame(result_tabControl, width=320, height=480)

        # Make today bar chart
        self.figure = plt.Figure(figsize=(chart_width, chart_height), dpi=100)
        self.figure_canvas = FigureCanvasTkAgg(self.figure, self.today_results_tab)
        self.axes = self.figure.add_subplot()
        self.axes.set_title('Mental Health Analysis')
        self.axes.bar(journalEntry.mha_values['categories'], journalEntry.mha_values['values'], width=0.7)
        self.figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.Y, expand=1)

        # Set label rotation
        self.axes.set_xticklabels(self.axes.get_xticklabels(), rotation=25, ha='right')

        # Month MHA Results
        self.month_results_tab = tk.Frame(result_tabControl, width=320, height=480)

        # Make month bar chart
        self.month_figure = plt.Figure(figsize=(chart_width, chart_height), dpi=100)
        self.month_figure_canvas = FigureCanvasTkAgg(self.month_figure, self.month_results_tab)
        self.month_axes = self.month_figure.add_subplot()
        self.month_axes.set_title('Mental Health Analysis')
        self.month_axes.bar(journal.month_avg_mha_values['categories'], journal.month_avg_mha_values['values'],
                            width=0.7)
        self.month_figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.Y, expand=1)

        # Set label rotation
        self.month_axes.set_xticklabels(self.month_axes.get_xticklabels(), rotation=25, ha='right')

        # Annual MHA Results
        self.year_results_tab = tk.Frame(result_tabControl, width=320, height=480)

        # Make annual bar chart
        self.year_figure = plt.Figure(figsize=(chart_width, chart_height), dpi=100)
        self.year_figure_canvas = FigureCanvasTkAgg(self.year_figure, self.year_results_tab)
        self.year_axes = self.year_figure.add_subplot()
        self.year_axes.set_title('Mental Health Analysis')
        self.year_axes.bar(journal.annual_avg_mha_values['categories'], journal.annual_avg_mha_values['values'],
                           width=0.7)
        self.year_figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.Y, expand=1)

        # Set label rotation
        self.year_axes.set_xticklabels(self.year_axes.get_xticklabels(), rotation=25, ha='right')

        # Add tabs
        result_tabControl.add(self.today_results_tab, text='Today')
        result_tabControl.add(self.month_results_tab, text='Month')
        result_tabControl.add(self.year_results_tab, text='Year')

        # Disclaimer Label
        disclaimerLabel = ttk.Label(self,
                                    text='Disclaimer: This tool is not a replacement for professional psychiatric '
                                         'and/or therapeutric assistance.')
        disclaimerLabel.place(x=130, y=window_height - 30)

        # Finalize widget
        result_tabControl.pack(expand=1, fill='both')

        # Configure ShowFrame event bindings
        self.update_journal(None)  # Update initial journal values
        self.bind('<<ShowFrame>>', self.update_journal)
        self.bind('<<ShowFrame>>', self.update_chart)
        self.bind('<<ShowFrame>>', self.update_recordings)
        self.bind('<<ShowFrame>>', self.reset_date)

    def reset_date(self, bindArgs):
        print('Reached reset_data()')
        # Decrement date
        self.date = self.date.now()

        # Update label
        self.date_label.config(text=self.date.strftime('%B %d, %Y'))
        self.date_label.update()

        # Update Journal
        journalEntry.loadEntry(int(self.date.strftime('%d')))
        self.update_journal(None)
        self.update_chart(None)  # Update Journal
        journalEntry.loadEntry(int(self.date.strftime('%d')))
        self.update_journal(None)
        self.update_chart(None)
        self.update_recordings(None)
        self.update_recordings(None)

    def move_date_back(self):
        print('Reached move_date_back()')
        # Decrement date
        self.date = self.date - timedelta(days=1)

        # Update label
        self.date_label.config(text=self.date.strftime('%B %d, %Y'))
        self.date_label.update()

        # Update Journal
        journalEntry.loadEntry(int(self.date.strftime('%d')))
        self.update_journal(None)
        self.update_chart(None)
        self.update_recordings(None)

    def move_date_forward(self):
        print('Reached move_date_forward()')
        # Increment date
        self.date = self.date + timedelta(days=1)

        # Update label
        self.date_label.config(text=self.date.strftime('%B %d, %Y'))
        self.date_label.update()

        # Update Journal
        journalEntry.loadEntry(int(self.date.strftime('%d')))
        self.update_journal(None)
        self.update_chart(None)
        self.update_recordings(None)

    def play_recording(self):
        print('Reached play_recording()')

        def start_playing():
            print('Reached start_playing()')
        # Get selected recording from listbox
        try:
            recording_index = self.recordings_box.curselection()
            filepath = self.recordings_box.get(recording_index)
        except Exception:
            print('No selected file.')
            pass
        # Construct full filepath
        filepath = f'{recording_path}/{filepath}'

        # Play the recording
        chunk = 1024  # Set chunk size of 1024 samples per data frame

        # Open the sound file
        wf = wave.open(filepath, 'rb')

        # Create an interface to PortAudio
        p = pyaudio.PyAudio()

        # Open a .Stream object to write the WAV file to
        # 'output = True' indicates that the sound will be played rather than recorded
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # Read data in chunks
        data = wf.readframes(chunk)

        # Play the sound by writing the audio data to the stream
        while data != '':
            stream.write(data)
            data = wf.readframes(chunk)

        # Close and terminate the stream
        stream.close()
        p.terminate()

    def stop_recording(self):
        print('Reached stop_recording()')

    def open_calendar_popup(self):
        print('Reached open_calendar_popup()')

        # Function to set new date based on calendar choice
        def setDate():
            # Assign new date
            newDateObj = datetime.datetime.strptime(cal.get_date(), '%m-%d-%Y')
            self.date = newDateObj

            # Update label
            self.date_label.config(text=self.date.strftime('%B %d, %Y'))
            self.date_label.update()

            # Update Journal
            journalEntry.loadEntry(int(self.date.strftime('%d')))
            self.update_journal(None)
            self.update_chart(None)
            self.update_recordings(None)

        # Find the center point and configure dimensions
        cal_window_width, cal_window_height = 220, 240
        cal_center_x = int(screen_width / 2 - cal_window_width / 2)
        cal_center_y = int(screen_height / 2 - cal_window_height / 2)

        # Create popup
        top = Toplevel(self)
        top.geometry(f'{cal_window_width}x{cal_window_height}+{cal_center_x}+{cal_center_y}')
        top.title('DatePicker')

        # Add calendar to popup
        day = int(self.date.today().strftime('%d'))
        month = int(self.date.today().strftime('%m'))
        year = int(self.date.today().strftime('%Y'))

        # Calendar Widget
        cal = Calendar(top, selectmode='day', year=year, month=month, day=day,
                       date_pattern="mm-dd-yyyy")
        cal.pack(padx=10, pady=10)

        # Select Date Button
        selDateBtn = ttk.Button(top, text='Confirm Date', command=setDate)
        selDateBtn.pack(padx=10, pady=10)

    def update_journal(self, bindArgs):
        print('Reached update_journal()')
        self.logs_text['state'] = 'normal'
        self.logs_text.delete('1.0', END)

        for i in range(1, len(journalEntry.session_log['dateTime'])):
            self.lineCount = self.lineCount + 1
            self.logs_text.insert(self.lineCount, '\n')  # Add space
            dateTime = journalEntry.session_log["dateTime"][i]
            dateTime.replace("'", '')
            formattedLogEntry = f'\n{dateTime} - ' \
                                f'{journalEntry.session_log["speaker"][i]}: ' \
                                f'{journalEntry.session_log["dialogue"][i]}'
            self.logs_text.insert(self.lineCount, formattedLogEntry)
        self.logs_text.delete('1.0', '2.0')  # Remove inital gap
        self.logs_text['state'] = 'disabled'

    def update_recordings(self, bindArgs):
        print('Reached update_recordings()')

        # Get date string from current date
        date_str = self.date.strftime('%m_%d_%Y')

        # Update filelist
        self.audio_file_list = os.listdir(recording_path)  # Get list of audio files

        # Delete existing info
        self.recordings_box.delete(0, END)

        # Populate the recordings box
        for item in self.audio_file_list:
            # Check if recording is dated for current date
            if date_str in item:
                self.recordings_box.insert(END, item)

    def update_chart(self, bindArgs):
        print('Reached update_chart()')

        # Replot data with new values
        print(f'MHA Values: {journalEntry.mha_values}')

        # Reconfigure today for new data
        self.axes.clear()
        self.axes.bar(journalEntry.mha_values['categories'], journalEntry.mha_values['values'], width=0.7)
        self.axes.set_title('Mental Health Analysis')
        self.axes.set_xticklabels(self.axes.get_xticklabels(), rotation=25, ha='right')
        self.figure_canvas.draw()

        # Reconfigure month for new data
        self.month_axes.clear()
        self.month_axes.bar(journal.month_avg_mha_values['categories'], journal.month_avg_mha_values['values'],
                            width=0.7)
        self.month_axes.set_title('Mental Health Analysis')
        self.month_axes.set_xticklabels(self.month_axes.get_xticklabels(), rotation=25, ha='right')
        self.month_figure_canvas.draw()

        # Reconfigure year for new data
        self.year_axes.clear()
        self.year_axes.bar(journal.annual_avg_mha_values['categories'], journal.annual_avg_mha_values['values'],
                           width=0.7)
        self.year_axes.set_title('Mental Health Analysis')
        self.year_axes.set_xticklabels(self.year_axes.get_xticklabels(), rotation=25, ha='right')
        self.year_figure_canvas.draw()


# Start the program
if __name__ == "__main__":
    print('Launching ESAI...')

    # Setup TTS
    tts = TTS()
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

    # Make directories if they don't exist
    try:
        recording_path = os.path.join(os.curdir, 'UserData/audio_recordings')
        journal_path = os.path.join(os.curdir, 'UserData/journals')

        # Audio Recordings
        if not os.path.exists(recording_path):
            os.makedirs(recording_path)
            print('Successfully created audio_recording directory.')

        # Journals
        if not os.path.exists(journal_path):
            os.makedirs(journal_path)
            print('Successfully created journals directory.')
    except OSError as error:
        print(f'Unable to create required directories. Error: {error}')

    # Load / Create Journal for the month
    currentJournalDate = datetime.datetime.now().strftime('%m-%Y')
    path = r'UserData/journals/%s.obj' % currentJournalDate
    try:
        # Load journal from system
        filehandler = open(path, 'rb')
        journal = pickle.load(filehandler)
        filehandler.close()

        # Update average mha_values
        journal.update_monthly_mha_avgs()
        journal.update_annual_mha_avgs()

        # Export updated journal
        journal.exportJournal()
        print(f'Successfully loaded journal for {currentJournalDate}')
    except FileNotFoundError:
        print('journal.obj not found. Creating new journal')
        journal = Journal()
        journal.exportJournal()

    # Load / create JournalEntry for the day
    journalEntry = JournalEntry()

    entryDate = int(datetime.datetime.now().strftime('%d'))
    journalEntry.loadEntry(entryDate)

    # Setup window dimensions
    window_width = 870
    window_height = 640

    # Global user_input field to track all user_input. Text gets added normally, audio needs converted to text first
    user_input = ''
    public_font = ("Arial", 12, "normal")

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
