# Imports
import tkinter as tk
from tkinter import ttk
import sv_ttk
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


class MentalHealthAnalyzer():
    def __init__(self, *args, **kwargs):
        self.text_clf = self.load_classifier()

    def load_classifier(self):
        # Attempt to load existing model. If model isn't found, create a new one
        nb_filename = 'res/classification_data/models/nb.sav'
        try:
            print('Attempting to load nb.sav...')
            self.text_clf = joblib.load(nb_filename)
            print('Successfully Loaded nb.sav')
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
            # text_clf = text_clf.fit(df.selftext, df.category)

            # Test Performance of NB Classifier
            # test_naive_bayes_classifier(text_clf, df)

            # Save model
            joblib.dump(text_clf, nb_filename)

            return text_clf
    def analyzeText(self, input_text):
        return self.text_clf.predict_proba([input_text])
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
        self.homeBtn = tk.PhotoImage(file='res/img/home.png').subsample(15, 15)

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
        for F in (HomePage, TextSessionPage, VoiceSessionPage, CopingPage, ResultsPage):
            frame = F(container, self)

            # Setup window dimensions
            window_width = 670

            # light/dark mode toggle
            uiToggle = ttk.Button(self, image=self.uiToggleImg, width=10, command=lambda: uiMode())
            uiToggle.place(x=window_width - 70, y=70)

            # home button
            homeBtn = ttk.Button(self, image=self.homeBtn, width=10, command=lambda: self.show_frame(HomePage))
            homeBtn.place(x=window_width - 70, y=10)

            # initializing frame of that object from each page planned
            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        # Show the home page
        self.show_frame(HomePage)

    # to display the current frame passed as parameter to switch to desired frame of program
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


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
                                    command=lambda: controller.show_frame(TextSessionPage))
        # textSessionBtn.grid(row=2, column=1, ipadx=50, ipady=50, padx=10, pady=10)
        textSessionBtn.pack(ipady=20, padx=10, pady=10)

        # Voice Session
        voiceSessionBtn = ttk.Button(body_frame, width=40, text="Start Voice Session",
                                     command=lambda: controller.show_frame(VoiceSessionPage))
        # voiceSessionBtn.grid(row=2, column=2, ipadx=50, ipady=50, padx=10, pady=10)
        voiceSessionBtn.pack(ipady=20, padx=10, pady=10)

        # Coping/De-stressing Exercises Activities
        copingPageBtn = ttk.Button(body_frame, width=40, text="Coping Exercises",
                                   command=lambda: controller.show_frame(CopingPage))
        copingPageBtn.pack(ipady=20, padx=10, pady=10)
        # copingPageBtn.grid(row=2, column=3, ipadx=50, ipady=50, padx=10, pady=10)

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

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        # self.bind('<Return>', self.enterPressed)

        # Setup window dimensions
        window_width = 670
        window_height = 440

        # Body Frame
        body_frame = ttk.Frame(self, width=window_width)
        body_frame.place(x=30, y=30)

        # Text Widget to Display Chat with Scrollbar
        output = tk.Text(body_frame, width=60, height=20)
        scrollBar = ttk.Scrollbar(body_frame, orient='vertical', command=output.yview)
        scrollBar.grid(column=1, row=0, sticky='nwes')
        output['yscrollcommand'] = scrollBar.set
        output.grid(column=0, row=0, sticky='ns')

        # Trigger initial output from Sai
        self.lineCount = 1.0
        output['state'] = 'normal'  # Re-enable editing to use insert()
        output.insert(self.lineCount, 'Sai: Welcome to ESAI. My name is Sai and I am here to '
                                      'provide you with emotional support as'
                                      ' needed. How are you feeling today?\n')
        output['state'] = 'disabled'  # Prevent user from editing output text

        # Footer Frame
        footer_frame = ttk.Frame(self, width=window_width)
        footer_frame.place(x=30, y=window_height - 70)

        # Entry Field
        input_field = ttk.Entry(footer_frame, width=60)
        input_field.pack(side='left', padx=5, pady=10)
        input_field.focus_set()  # Bring user to text field immediately

        # Submit Button
        submitBtn = ttk.Button(footer_frame, text='Submit', width=8,
                               command=lambda: setOutput(input_field.get()))
        submitBtn.pack(side='right', padx=5, pady=10)

        def setOutput(inputText):
            input_field.delete(0, 'end')  # Erase input field
            # Validate inputText is not null before continuing
            if len(inputText) >= 1:
                # Set User Output
                output['state'] = 'normal'  # Re-enable editing to use insert()
                self.lineCount = self.lineCount + 1
                output.insert(self.lineCount, ('You: ' + inputText + "\n"))
                output['state'] = 'disabled'  # Prevent user from editing output text

                # Append session logs
                self.session_log['speaker'].append('You')
                self.session_log['dialogue'].append(inputText)

                # Set Sai's Output
                output['state'] = 'normal'  # Re-enable editing to use insert()
                self.lineCount = self.lineCount + 1
                response = self.getResponse(inputText)
                output.insert(self.lineCount, response + "\n")
                output['state'] = 'disabled'  # Prevent user from editing output text

                # Append session logs
                self.session_log['speaker'].append('Sai')
                self.session_log['dialogue'].append(inputText)

    # Get response based on the users input and return it to be printed under Sai's response
    def getResponse(self, inputText):
        return mha.analyzeText(inputText)


class VoiceSessionPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)


class CopingPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)


class ResultsPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)


# Start the program
if __name__ == "__main__":
    print('Launching ESAI...')

    # Setup MainApp
    darkUI = False
    app = MainApp()
    app.title("ESAI: An Emotional Support AI")

    # Setup tkinter theme
    sv_ttk.set_theme("light")

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
