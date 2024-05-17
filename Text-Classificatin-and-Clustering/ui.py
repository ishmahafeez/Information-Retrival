import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from threading import Thread
from main import preprocess_query, classify_new_query, knn_classifier, count_vectorizer, tfidf_transformer, train_knn_classifier

def classify_query():
    # Get query from user
    query = query_entry.get()

    # Classify the query
    predicted_label = classify_new_query(query, knn_classifier, count_vectorizer, tfidf_transformer)

    # Clear results
    results_text.delete('1.0', tk.END)

    # Display predicted label
    results_text.insert(tk.END, "Predicted Label for the Query:\n", "header")
    results_text.insert(tk.END, f"{predicted_label}\n", "result")

def upload_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        with open(file_path, 'r') as file:
            query_entry.delete(0, tk.END)
            query_entry.insert(0, file.read())

def run_gui():
    root = tk.Tk()
    root.title("Text Classification System")
    root.configure(bg="#f2f2f2")

    # Header Frame
    header_frame = tk.Frame(root, bg="#4CAF50", pady=10)
    header_frame.pack(fill="x")

    header_label = tk.Label(header_frame, text="Enter your query:", bg="#4CAF50", fg="white", font=("Arial", 14, "bold"))
    header_label.pack(side="left", padx=20)

    # Query Input
    global query_entry
    query_entry = ttk.Entry(root, width=50, font=("Arial", 12))
    query_entry.pack(pady=(20, 5))

    # Button
    classify_button = ttk.Button(root, text="Classify Query", command=classify_query, style="Custom.TButton")
    classify_button.pack(pady=10)

    # Upload Button
    upload_button = ttk.Button(root, text="Upload File", command=upload_file, style="Custom.TButton")
    upload_button.pack(pady=5)

    # Result Display
    global results_text
    results_text = scrolledtext.ScrolledText(root, width=50, height=10, wrap=tk.WORD, font=("Arial", 12), bg="white")
    results_text.pack(pady=10)

    results_text.tag_configure("header", font=("Arial", 12, "bold"))
    results_text.tag_configure("result", foreground="#1E90FF")

    # Style
    style = ttk.Style()
    style.configure("Custom.TButton", foreground="white", background="#008CBA", font=("Arial", 12))

    root.mainloop()

gui_thread = Thread(target=run_gui)
gui_thread.start()
