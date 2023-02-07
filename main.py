# Imports
import queue
import pyttsx3
import sv_ttk
import threading
import joblib
import os
import tkinter as tk
from tkinter import ttk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.chat.util import Chat, reflections


class SaiChatBot:
    def __init__(self):
        # Reflections to handle basic input and related output
        self.reflections = {
            "i am": "you are",
            "i was": "you were",
            "i": "you",
            "i'm": "you are",
            "i'd": "you would",
            "i've": "you have",
            "i'll": "you will",
            "my": "your",
            "you are": "I am",
            "you were": "I was",
            "you've": "I have",
            "you'll": "I will",
            "your": "my",
            "yours": "mine",
            "you": "me",
            "me": "you"
        }

        # Create simple set of rules
        self.pairs = [
            [
                r"my name is (.*)",
                ["Hello %1, How are you today ?", ]
            ],
            [
                r"hi|hey|hello",
                ["Hello", "Hey there", ]
            ],
            [
                r"what is your name ?",
                ["My name is Sai, and I am your Emotional Support AI", ]
            ],
            [
                r"how are you ?",
                ["I'm doing great. How are you feeling today?", ]
            ],
            [
                r"sorry (.*)",
                ["Its alright", "Its OK, never mind", ]
            ]
        ]

    # Start chatting
    def chat(self, input_text):
        print('Reached chat():')

        # Compile pairs and reflections
        chat = Chat(self.pairs, self.reflections)

        # Get response
        return chat.respond(input_text)


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


class MentalHealthAnalyzer:
    def __init__(self, *args, **kwargs):
        self.text_clf = self.load_classifier()

    def load_classifier(self):
        # Attempt to load existing model. If model isn't found, create a new one
        nb_filename = 'res/classification_data/models/nb.sav'
        try:
            print('Attempting to load nb.sav...')
            return joblib.load(nb_filename)
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

        # Setup window dimensions
        window_width = 670
        window_height = 440

        # Body Frame
        body_frame = ttk.Frame(self, width=window_width)
        body_frame.place(x=30, y=30)

        # Text Widget to Display Chat with Scrollbar
        self.output = tk.Text(body_frame, width=65, height=20)
        scrollBar = ttk.Scrollbar(body_frame, orient='vertical', command=self.output.yview)
        scrollBar.grid(column=1, row=0, sticky='nwes')
        self.output['yscrollcommand'] = scrollBar.set
        self.output.grid(column=0, row=0, sticky='ns')

        # Trigger initial output from Sai
        self.lineCount = 1.0
        self.output['state'] = 'normal'  # Re-enable editing to use insert()
        starter_text = 'Sai: Welcome to ESAI. My name is Sai and I am here to ' \
                       'provide you with emotional support as ' \
                       'needed. How are you feeling today?\n'
        self.output.insert(self.lineCount, starter_text)
        self.output['state'] = 'disabled'  # Prevent user from editing output text
        tts_queue.put(starter_text)

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

    def setOutput(self, bindArg): # bindArg acts as a 2nd parameter to allow enter key to send input
        inputText = self.input_field.get() # Get input text and store before erasing
        self.input_field.delete(0, 'end')  # Erase input field
        # Validate inputText is not null before continuing
        if len(inputText) >= 1:
            # Set User Output
            self.output['state'] = 'normal'  # Re-enable editing to use insert()
            self.lineCount = self.lineCount + 1
            self.output.insert(self.lineCount, ('You: ' + inputText + "\n"))
            self.output['state'] = 'disabled'  # Prevent user from editing output text

            # Append session logs
            self.session_log['speaker'].append('You')
            self.session_log['dialogue'].append(inputText)

            # Set Sai's Output
            self.output['state'] = 'normal'  # Re-enable editing to use insert()
            self.lineCount = self.lineCount + 1
            response = self.getResponse(inputText)
            self.output.insert(self.lineCount, response + "\n")
            self.output['state'] = 'disabled'  # Prevent user from editing output text
            tts_queue.put(response[:5])  # the :5 is to prevent tts from including sai's name

            # Append session logs
            self.session_log['speaker'].append('Sai')
            self.session_log['dialogue'].append(inputText)

    # Get response based on the users input and return it to be printed under Sai's response
    def getResponse(self, inputText):
        # Check to see if any flags are triggered (chance of disorder > 50%)

        # Get chatbot response
        try:
            response = 'Sai: ' + sai_bot.chat(inputText)
        except:
            response = "Sai: I'm sorry, I do not understand."

        return response


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

    # Setup TTS
    tts_queue = queue.Queue()
    print('TTS Queue Created.')

    tts_thread = TTSThread(tts_queue)
    print('TTS Thread Started.')

    # Setup Chat Bot
    sai_bot = SaiChatBot()

    # Setup MainApp
    darkUI = False
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
