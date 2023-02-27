# Imports
import csv
import queue

import numpy as np
import pyttsx3
import spacy
import sv_ttk
import threading
import joblib
import os
import unidecode
import contractions
import re
import pandas as pd
import tkinter as tk
import speech_recognition as sr
from tkinter import ttk
from tkinter import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tkVideoPlayer import TkinterVideo


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
            print('Setting-Up Naive Bayes Classifier...')

            if not os.path.exists('res/classification_data/models'):
                os.makedirs('res/classification_data/models')

            # Setup NB Classification Pipeline
            saiBot = Pipeline([('vect', CountVectorizer()),
                               ('tfidf', TfidfTransformer()),
                               ('clf', MultinomialNB())])
            print("Features Extracted.")
            print("Term Frequencies Extracted.")
            print('Naive Bayes Classifier Setup Complete.')

            # Load datasets
            df = self.load_data()

            print(df.head())
            # Run Naive Bayes(NB) ML Algorithm to build model
            saiBot = saiBot.fit(df.prompt, df.response)

            # Test Performance of NB Classifier
            # test_naive_bayes_classifier(text_clf, df)

            # Save model
            joblib.dump(saiBot, sai_bot_path)

            return saiBot

    def load_data(self):
        # Configure filepath
        data_filepath = 'res/classification_data/datasets/SaiBotData.csv'

        # Try to load existing master dataset. If not found, return error.
        try:
            return pd.read_csv(data_filepath)
        except FileNotFoundError:
            print('SaiBotData.csv is missing. Please insert SaiBotData.csv into the '
                  '"res/classification_data/datasets" folder')

    def get_response(self, input_text):
        response = self.saiBot.predict([input_text])
        print(f'Response: {response}')

        # Check if response is valid - Valid >= 30% chance
        proba = self.saiBot.predict_proba([input_text])
        for p in proba[0]:
            print(p)
            if p > .2:
                print(f'Valid - {input_text}')
                return response[0]
            else:
                print(f'Invalid - {input_text}')
                return None
        #proba = np.array(proba[0])
        #p#rint(f'Probability: {proba}')
        #if proba.any() > 0.4:



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
                with sr.Microphone() as mic:
                    print('Listening...')
                    self.recognizer.adjust_for_ambient_noise(mic)  # Filter out background noise
                    audio = self.recognizer.record(source=mic, duration=5)  # Initialize input from mic

                    self.text = self.text + self.recognizer.recognize_google(audio)  # Convert audio to text
                    print(f'Voice Input: {self.text}')
            except:
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


class MentalHealthAnalyzer:
    def __init__(self, *args, **kwargs):
        self.text_clf = self.load_classifier()

    def load_classifier(self):
        # Attempt to load existing model. If model isn't found, create a new one
        nb_filename = 'res/classification_data/models/nb.sav'
        try:
            print('Attempting to load nb.sav...')
            text_clf = joblib.load(nb_filename)
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

            # Load datasets
            df = self.load_data()
            print(df.head())
            # Run Naive Bayes(NB) ML Algorithm to build model
            print('Fitting Model...')
            text_clf = text_clf.fit(df.selftext.values.astype('U'), df.category.values.astype('U'))
            print('Model has been fitted.\nSaving model to res/classification_data/models/nb.sav')
            # Test Performance of NB Classifier
            # test_naive_bayes_classifier(text_clf, df)

            # Save model
            joblib.dump(text_clf, nb_filename)
            print('Model saved.')

            return text_clf

    def text_cleanup(self, text):
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

    def load_data(self):
        # Configure Filepaths
        master_filepath = 'res/classification_data/datasets/master-set.csv'

        try:  # Try to load existing master dataset. If not found, create new one
            return pd.read_csv(master_filepath)
        except FileNotFoundError:

            filepath_dict = {'anxiety': 'res/classification_data/datasets/20k/anxiety.csv',
                             'depression': 'res/classification_data/datasets/20k/depression.csv',
                             'tourettes': 'res/classification_data/datasets/20k/tourettes.csv',
                             'suicide': 'res/classification_data/datasets/20k/suicidewatch.csv',
                             'adhd': 'res/classification_data/datasets/20k/adhd.csv',
                             'schizophrenia': 'res/classification_data/20k/datasets/schizophrenia.csv',
                             'eatingdisorder': 'res/classification_data/20k/datasets/eatingdisorder.csv',
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

            for i in range(0, len(df['selftext'])):
                # Get the value of current selftext
                value = df['selftext'].iloc[i]

                # Clean the data
                value = self.text_cleanup(value)

                # Update the dataframe for master-set
                df['selftext'].iloc[i] = value
                print(df['selftext'].iloc[i])

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
                HomePage, TextSessionPage, VentingPage, CopingPage, ResultsPage, BreathingActivity,
                IdentifyingSurroundings):
            frame = F(container, self)

            # Setup window dimensions
            window_width = 670

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
        window_width = 670
        window_height = 440

        # Body Frame
        body_frame = ttk.Frame(self, width=window_width, height=window_height - 400)
        body_frame.pack(side="top", pady=75, fill="x", expand=True)
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


class TextSessionPage(ttk.Frame):
    # Class Fields
    lineCount = 0  # used to make sure new outputs go onto next line
    session_log = {
        'speaker': [],
        'dialogue': []
    }
    starter_text = 'Sai: Welcome to ESAI. My name is Sai and I am here to ' \
                   'provide you with emotional support as ' \
                   'needed. How are you feeling today?\n'

    user_input = ''

    new_questions_path = 'res/classification_data/datasets/new_questions.csv'
    new_questions = [[]]

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)

        # Load new questions file so we can edit during program
        print('Attempting to load new_questions.csv')
        try:
            with open(self.new_questions_path, 'rt') as f:
                reader = csv.reader(f)
                self.new_questions = list(reader)
                print('Loaded new_questions.csv')
        except FileNotFoundError:
            with open(self.new_questions_path, 'w+') as f:
                write = csv.writer(f)
                write.writerow(self.new_questions)
            print('Created new_questions.csv')

        def jumpToResults():
            print('Reached jumpToResults().')
            # Get user input blob of text
            input_text = self.user_input

            # Jump to Results Page
            resultsPage = ResultsPage(parent, controller)
            resultsPage.set_input_text(input_text)  # Pass / Set the input_text to results page
            print(input_text)

            controller.show_frame(ResultsPage)

        # Setup window dimensions
        window_width = 670
        window_height = 440

        # Body Frame
        body_frame = ttk.Frame(self, width=window_width)
        body_frame.place(x=30, y=30)

        # Results button
        # https://www.flaticon.com/free-icons/notepad created by Freepik - Flaticon
        self.resultsBtnImg = tk.PhotoImage(file='res/img/results.png').subsample(15, 15)
        resultsBtn = ttk.Button(self, image=self.resultsBtnImg, width=10, command=jumpToResults)
        resultsBtn.place(x=window_width - 70, y=190)

        # Text Widget to Display Chat with Scrollbar
        self.output = tk.Text(body_frame, width=65, height=20)
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
        self.input_field = ttk.Entry(footer_frame, width=60)
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
            self.session_log['speaker'].append('You')
            self.session_log['dialogue'].append(inputText)

            # Set Sai's Output
            self.output['state'] = 'normal'  # Re-enable editing to use insert()
            self.lineCount = self.lineCount + 1
            response = self.getResponse(inputText)
            self.output.insert(self.lineCount, response + "\n")
            self.output['state'] = 'disabled'  # Prevent user from editing output text
            tts_queue.put(response[5:])  # the :5 is to prevent tts from including sai's name

            # Append session logs
            self.session_log['speaker'].append('Sai')
            self.session_log['dialogue'].append(response[5:])
            print(f'\nSession Log: {self.session_log.items()}')

    # Get response based on the users input and return it to be printed under Sai's response
    def getResponse(self, inputText):
        # Check to see if any flags are triggered (chance of disorder > 50%)
        disorders = mha.text_clf.classes_.tolist()

        report = mha.analyze_text(self.user_input)
        print(f'\nDetailed Analysis: {report}')

        for i in range(0, len(report)):
            print(report[i])
            if report[i] > 50:  # If there is a 30% or higher chance of illness, add it to detected illness list
                print(f'\n{disorders[i]} Detected.')
            else:
                print(f'\n{disorders[i]}: Not Detected.')

        # Get Sai's response
        response = sai_bot.get_response(inputText)
        print(f'Response: {response}')
        if response is not None:
            return 'Sai: ' + response
        else:
            try:  # Save the unknown question/comment, so it can be added later
                self.new_questions.append(inputText)
                print('Appended new question/query.')

                with open(self.new_questions_path, 'w+') as f:
                    write = csv.writer(f)
                    write.writerow(self.new_questions)
                print('Saved question to csv.')
            except Exception as e:
                print(e)
            return "Sai: I'm sorry, I do not understand."


class VentingPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        print('Listening...')
        self.controller = controller

        # variable to store input from speech
        self.input_text = None

        def button_press():
            # trigger stt
            stt.toggle()  # shouldn't return anything because stt.listening = False at this time.

            def jumpToResults():
                print('Reached jumpToResults().')
                self.input_text = stt.toggle()  # Should return voice input now that stt.listening was = True

                resultsPage = ResultsPage(parent, controller)
                resultsPage.set_input_text(self.input_text)  # Pass / Set the input_text to results page
                print(self.input_text)

                self.controller.show_frame(ResultsPage)

                print(self.input_text)

            # Reconfigure button to reflect stopping the session
            self.startListeningBtn.config(text='End Session', command=jumpToResults)
            self.startListeningBtn.pack()

        # Configure listening button
        self.startListeningBtn = ttk.Button(self, text='Start Venting Session', command=button_press)
        self.startListeningBtn.pack()


class CopingPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        # Setup window dimensions
        window_width = 670
        window_height = 440

        # Body Frame
        body_frame = ttk.Frame(self, width=window_width, height=window_height - 400)
        body_frame.pack(side="top", pady=75, fill="x", expand=True)
        body_frame.anchor('center')

        # Buttons to choose session type
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

        # Setup variables
        self.instruction = 'Please press the "Start" button and close your eyes to begin the breathing activity.'

        # Label to store instructions
        self.instruction_label = ttk.Label(self, text=self.instruction)
        self.instruction_label.pack()

        # Setup the thread for the activity to prevent tkinter from freezing
        self.breathing_thread = threading.Thread(target=self.start_breathing)

        # Button to start activity thread
        start_button = ttk.Button(self, text="Start", command=self.start_thread)
        start_button.pack()

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
            print(f'Breathing Round {i+1}/5')
            self.after(4000, breathe_in())
            self.after(4000, hold_breathe())
            self.after(4000, breathe_out())
            self.after(4000, hold_breathe())


        # End activity
        activity_status = 'Breathing activity completed'
        print(activity_status)
        self.instruction_label.config(text=activity_status)
        tts_queue.put('Great job! You have finished the breathing activity. '
                      'I hope this helped you de-stress at least a little bit.')

    def start_activity(self, bindArgs):
        # Send instruction to tts_queue for speech
        tts_queue.put(self.instruction)

        # Bind the end of video to the loopVideo function
        # self.bind("<<ShowFrame>>", breathing_animation.run)


class IdentifyingSurroundings(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        print('Reached IdentifyingSurroundings.')


class ResultsPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        print('Reached ResultsPage.')

        # Set variable to store text to analyze
        self.input_text = None

        # Results Label
        resultsLabel = ttk.Label(self, text='Results Page')
        resultsLabel.pack()

        print(self.input_text)

    # Function to set text blob
    def set_input_text(self, input_text):
        self.input_text = input_text


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

    # Setup Chat Bot
    sai_bot = SaiBot()

    # Setup MainApp
    darkUI = True
    app = MainApp()
    app.title("ESAI: An Emotional Support AI")

    # Setup tkinter theme
    sv_ttk.set_theme("dark")

    # Load MentalHealthAnalyzer / Naive Bayes Classifier
    mha = MentalHealthAnalyzer()

    # Setup window dimensions
    window_width = 670
    window_height = 440

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
