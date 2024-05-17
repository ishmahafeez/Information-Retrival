import tkinter as tk
from tkinter import ttk, filedialog
from main import preprocess_query, classify_new_query, knn_classifier, count_vectorizer, tfidf_transformer, accuracy, report, evaluate_clustering, purity, silhouette, rand_index, dict

def classify_query():
    # Get the query from the entry widget
    query = query_entry.get()
    # Classify the query using the classifier
    predicted_label = classify_new_query(query, knn_classifier, count_vectorizer, tfidf_transformer)
    # Display the predicted label
    result_label.config(text=f"Predicted Label for the Query: {predicted_label}")
    # Display the accuracy
    accuracy_label.config(text=f"Accuracy: {accuracy}")
    # Display the classification report
    report_text.config(state=tk.NORMAL)
    report_text.delete('1.0', tk.END)
    report_text.insert(tk.END, report)
    report_text.config(state=tk.DISABLED)

def upload_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        with open(file_path, 'r') as file:
            query_entry.delete(0, tk.END)
            query_entry.insert(0, file.read())

# Create the main window
root = tk.Tk()
root.title("Text Classification")

# Set the style
style = ttk.Style()
style.configure("Custom.TLabel", font=("Arial", 12), background="#f0f0f0")
style.configure("Custom.TButton", font=("Arial", 12),background="#f0f0f0")

# GUI elements
query_label = ttk.Label(root, text="Enter your query:", style="Custom.TLabel")
query_entry = ttk.Entry(root, width=50)
classify_button = ttk.Button(root, text="Classify", command=classify_query, style="Custom.TButton")
upload_button = ttk.Button(root, text="Upload File", command=upload_file, style="Custom.TButton")
result_label = ttk.Label(root, text="", style="Custom.TLabel")
accuracy_label = ttk.Label(root, text="", style="Custom.TLabel")
report_label = ttk.Label(root, text="Classification Report:", style="Custom.TLabel")
report_text = tk.Text(root, wrap=tk.WORD, width=80, height=15)
report_text.config(state=tk.DISABLED)

# Layout
query_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
query_entry.grid(row=0, column=1, columnspan=2, padx=10, pady=(10, 5), sticky="ew")
classify_button.grid(row=1, column=0, padx=10, pady=5)
upload_button.grid(row=1, column=1, padx=10, pady=5)
result_label.grid(row=2, column=0, columnspan=3, padx=10, pady=5)
accuracy_label.grid(row=3, column=0, columnspan=3, padx=10, pady=5)
report_label.grid(row=4, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")
report_text.grid(row=5, column=0, columnspan=3, padx=10, pady=(0, 10))

# Add a function to display clustering report
def display_clustering_report():
    # Call evaluate_clustering function and get the metrics
    purity, silhouette, rand_index = evaluate_clustering(dict)
    # Display clustering report
    clustering_report_label.config(text=f"Clustering Report:\n\nPurity: {purity}\nSilhouette Score: {silhouette}\nRandom Index: {rand_index}", style="Custom.TLabel")

# GUI elements for clustering report
clustering_report_label = ttk.Label(root, text="", style="Custom.TLabel")
clustering_report_button = ttk.Button(root, text="Display Clustering Report", command=display_clustering_report, style="Custom.TButton")

# Layout for clustering report
clustering_report_label.grid(row=6, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")
clustering_report_button.grid(row=7, column=0, columnspan=3, padx=10, pady=5)

root.mainloop()
