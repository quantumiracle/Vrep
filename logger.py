import time
import threading
import logging
import Tkinter as tk # Python 2.x
import ScrolledText

class TextHandler(logging.Handler):
    # This class allows you to log to a Tkinter Text or ScrolledText widget
    # Adapted from Moshe Kaplan: https://gist.github.com/moshekaplan/c425f861de7bbf28ef06

    def __init__(self, text):
        # run the regular Handler __init__
        logging.Handler.__init__(self)
        # Store a reference to the Text it will log to
        self.text = text

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text.configure(state='normal')
            self.text.insert(tk.END, msg + '\n')
            self.text.configure(state='disabled')
            # Autoscroll to the bottom
            self.text.yview(tk.END)
        # This is necessary because we can't modify the Text from other threads
        self.text.after(0, append)

class myGUI(tk.Frame):

    # This class defines the graphical user interface

    def __init__(self, parent, title, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.build_gui(title)

    def build_gui(self,title):
        # Build GUI
        self.root.title(title)
        self.root.option_add('*tearOff', 'FALSE')
        self.grid(column=0, row=0, sticky='ew')
        self.grid_columnconfigure(0, weight=1, uniform='a')
        self.grid_columnconfigure(1, weight=1, uniform='a')
        self.grid_columnconfigure(2, weight=1, uniform='a')
        self.grid_columnconfigure(3, weight=1, uniform='a')

        # Add text widget to display logging info
        st = ScrolledText.ScrolledText(self, state='disabled')
        st.configure(font='TkFixedFont')
        st.grid(column=0, row=1, sticky='w', columnspan=4)
        self.root.configure(height=200, width=1200)

        # Create textLogger
        text_handler = TextHandler(st)

        # Logging configuration
        logging.basicConfig(level=logging.INFO)

        # Add the handler to logger
        logger = logging.getLogger()
        logger.addHandler(text_handler)

class Printer:
    def __init__(self,title):
        def launch_tkinter_screen():
            root = tk.Tk()
            myGUI(root,title)
            root.mainloop()
        screen_thread = threading.Thread(target=launch_tkinter_screen,args=[])
        screen_thread.daemon = True
        screen_thread.start()

    def print_to_screen(self, msg):
        logging.info(msg)
