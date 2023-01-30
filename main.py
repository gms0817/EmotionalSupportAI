import tkinter as tk
from tkinter import ttk


class MainApp(tk.Tk):
    # init function for MainApp
    def __init__(self, *args, **kwargs):
        # init function for Tk
        tk.Tk.__init__(self, *args, **kwargs)

        # Create a container
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # initializes frames array
        self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (HomePage, SessionChoicePage, TextSessionPage, VoiceSessionPage, CopingPage, ResultsPage):
            frame = F(container, self)

            # initializing frame of that object from each page planned
            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")
        # Show the home page
        self.show_frame(HomePage)

    # to display the current frame passed as parameter to switch to desired frame of program
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # Buttons to choose session type
        textSessionBtn = ttk.Button(self, text="Start Text Session",
                                    command=lambda: controller.show_frame(TextSessionPage))
        textSessionBtn.grid(row=1, column=1)

        voiceSessionBtn = ttk.Button(self, text="Start Voice Session,",
                                     command=lambda: controller.show_frame(VoiceSessionPage))
        voiceSessionBtn.grid(row=1, column=2)


class SessionChoicePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)


class TextSessionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)


class VoiceSessionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)


class CopingPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)


class ResultsPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)


# Start the program
if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
