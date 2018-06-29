import tkinter as tk
from tkinter.messagebox import showinfo
import re
import json
import cv2

from main import addNewPerson, identifyPeople

def open_capture(address):
    try:
        vid = cv2.VideoCapture('http://' + str(address) + '/video')
        if not vid.isOpened():
            popup_showinfo("Camera Error", "Wrong Address, Camera Connection Error")
            return None
    except cv2.error as e:
        print("cv2.error:", e)
    except Exception as e:
        print("Exception:", e)
    return vid


def delete_label(label):
    if not len(label) > 0:
        popup_showinfo("Label Error", 'Label Empty')
        return
    f = open('./facerec_128D.txt', 'r')
    data_set = json.loads(f.read());
    if label in data_set:
        del data_set[label]
    else:
        popup_showinfo("Label Error", 'Label not Found')
    f = open('./facerec_128D.txt', 'w');
    f.write(json.dumps(data_set))


def list_labels(textbox):
    f = open('./facerec_128D.txt', 'r')
    data_set = json.loads(f.read());
    edit_textbox(textbox, 'LABELS\n')
    for person in data_set.keys():
        edit_textbox(textbox, person + '\n', True)


def fill_instruction(textbox, filename):
    edit_textbox(textbox, '')
    with open(filename, 'r') as f:
        text = f.readlines()
        for t in text:
            edit_textbox(textbox, t, True)


def edit_textbox(textbox, text, append=False):
    textbox.configure(state='normal')
    if append:
        textbox.insert(tk.END, text)
    else:
        textbox.delete(1.0, tk.END)
        textbox.insert(tk.END, text)
    textbox.configure(state='disabled')


def popup_showinfo(winname, message):
    showinfo(winname, message)


def ip_regex(ip):
    patternIP = re.compile(
        "^[1-2]?[0-9]{1,2}\.[1-2]?[0-9]{1,2}\.[1-2]?[0-9]{1,2}\.[1-2]?[0-9]{1,2}:\d{1,5}$")  # "^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}$")
    valid = patternIP.match(ip)
    if valid:
        print("Valid IP")
        return True
    else:
        print("Invalid IP")
        return False


def learning_mode(textbox):
    address = addressEntry.get()
    newLabel = labelEntry.get()
    fill_instruction(textbox, 'learningManual.txt')
    if not len(newLabel) > 0:
        popup_showinfo("Label Error", 'Label Empty')
        return
    if not ip_regex(address):
        popup_showinfo("IP Error", "Invalid IP")
    else:
        capture = open_capture(address)
        if capture is not None:
            addNewPerson(capture, newLabel)


def detection_mode(textbox):
    address = addressEntry.get()
    fill_instruction(textbox, 'detectionManual.txt')
    if not ip_regex(address):
        popup_showinfo("IP Error", "Invalid IP")
    else:
        capture = open_capture(address)
        if capture is not None:
            identifyPeople(capture)


# LAYOUT
# Mode label        | Label Options Label
# Learning Button   | List Labels Buttton
# Detection Button  | Delete Labels Button
# Address Label     | Help Label
# Address Entry     | Display Intructions Button
# Label Label       |
# Label Entry       |
#             Main TextBox


# INIT
window = tk.Tk()
window.title("Face Recognition")

# window.geometry('200x300+100+100')

# MODE
modeLabel = tk.Label(window, text="Mode:")
modeLabel.grid(row=0, column=0, sticky=tk.W, padx=20, pady=(10, 0))

buttonLearning = tk.Button(window, text="Learning", width=20, command=lambda : learning_mode(mainTextBox))
buttonLearning.grid(row=1, column=0, sticky=tk.W, padx=20, pady=(0, 10))

buttonDetection = tk.Button(window, text="Detection", width=20, command=lambda :detection_mode(mainTextBox))
buttonDetection.grid(row=2, column=0, sticky=tk.W, padx=20, pady=(0, 10))

# ADDRESS
addressLabel = tk.Label(window, text="Address:")
addressLabel.grid(row=3, column=0, sticky=tk.W, padx=20, pady=(10, 0))

addressEntry = tk.Entry(window, width=25)
addressEntry.grid(row=4, column=0, sticky=tk.W, padx=20)
addressEntry.insert(tk.END, '192.168.137.177:8080')

# NEW DATA LABEL
labelLabel = tk.Label(window, text="Label:")
labelLabel.grid(row=5, column=0, sticky=tk.W, padx=20, pady=(10, 0))

labelEntry = tk.Entry(window, width=25)
labelEntry.grid(row=6, column=0, sticky=tk.W, padx=20, pady=(0, 20))

# LABEL OPTIONS
labelsLabel = tk.Label(window, text="Label Options:")
labelsLabel.grid(row=0, column=1, sticky=tk.W, padx=20, pady=(10, 0))

buttonDeleteLabel = tk.Button(window, text="Delete Label", width=20, command=lambda: delete_label(labelEntry.get()))
buttonDeleteLabel.grid(row=2, column=1, sticky=tk.W, padx=20, pady=(0, 10))

buttonLabels = tk.Button(window, text="List Labels", width=20, command=lambda: list_labels(mainTextBox))
buttonLabels.grid(row=1, column=1, sticky=tk.W, padx=20, pady=(0, 10))

# HELP & INSTRUCTION
mainTextBox = tk.Text(window, width=45, height=10, wrap=tk.WORD)
mainTextBox.grid(row=10, column=0, sticky=tk.W, columnspan=2, padx=10, pady=10)
mainTextBox.configure(state='disabled')

helpLabel = tk.Label(window, text="Help:")
helpLabel.grid(row=3, column=1, sticky=tk.W, padx=20, pady=(10, 0))

buttonManual = tk.Button(window, text="Display Intructions", width=20,
                         command=lambda: fill_instruction(mainTextBox, 'instruction.txt'))
buttonManual.grid(row=4, column=1, sticky=tk.W, padx=20, pady=(0, 10))

# ----------------

fill_instruction(mainTextBox, 'instruction.txt')

# ip_regex('10.10')
# ip_regex('192.168.137.218:8080')
# ip_regex('999.999.999.999:8080')

window.mainloop()
