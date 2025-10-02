import tkinter as tk
from tkinter import ttk
import threading
import subprocess
import random
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")  # Ensures compatibility with Tkinter
import matplotlib.pyplot as plt

class AIModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Model Trainer")
        
        self.parameters = {}
        self.create_main_page()
    
    def create_main_page(self):
        """Create parameter selection page."""
        self.frame_main = tk.Frame(self.root)
        self.frame_main.pack(fill=tk.BOTH, expand=True)
        
        self.param_entries = {}
        for i in range(20):
            tk.Label(self.frame_main, text=f"Param {i+1}:").grid(row=i, column=0, padx=10, pady=2)
            entry = tk.Entry(self.frame_main)
            entry.grid(row=i, column=1, padx=10, pady=2)
            self.param_entries[f"param_{i+1}"] = entry
        
        self.train_button = tk.Button(self.frame_main, text="Train Model", command=self.start_training)
        self.train_button.grid(row=21, column=0, columnspan=2, pady=10)
        
        self.test_button = tk.Button(self.frame_main, text="Test Model", command=self.start_testing)
        self.test_button.grid(row=22, column=0, columnspan=2, pady=10)
    
    def get_parameters(self):
        """Retrieve all user-defined parameters."""
        return {key: entry.get() for key, entry in self.param_entries.items()}
    
    def start_training(self):
        """Switch to training progress page and start subprocess."""
        self.parameters = self.get_parameters()
        self.switch_to_progress("Training")
        threading.Thread(target=self.train_model, daemon=True).start()
    
    def start_testing(self):
        """Switch to testing progress page and start subprocess."""
        self.parameters = self.get_parameters()
        self.switch_to_progress("Testing")
        threading.Thread(target=self.test_model, daemon=True).start()
    
    def switch_to_progress(self, mode):
        """Create progress page."""
        self.clear_frame()
        self.progress_label = tk.Label(self.root, text=f"{mode} in progress...")
        self.progress_label.pack()
        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=10)
    
    def train_model(self):
        """Mock training process."""
        for i in range(100):
            time.sleep(0.05)
            self.progress_bar["value"] = i+1
            self.root.update_idletasks()
        self.switch_to_results()
    
    def test_model(self):
        """Mock testing process."""
        # for i in range(100):
        #     time.sleep(0.05)
        #     self.progress_bar["value"] = i+1
        #     self.root.update_idletasks()
        self.switch_to_results()
    
    def switch_to_results(self):
        """Show random results with a graph."""
        # self.clear_frame()
        # tk.Label(self.root, text="Training/Testing Results").pack()
        
        # fig, ax = plt.subplots()
        # random_data = [random.uniform(0, 1) for _ in range(10)]
        # ax.plot(random_data, marker='o', linestyle='-', color='b')
        # ax.set_title("Random Model Performance")
        # ax.set_xlabel("Epochs")
        # ax.set_ylabel("Accuracy")
        # canvas = FigureCanvasTkAgg(fig, master=self.root)
        # canvas.get_tk_widget().pack()
        # canvas.draw()
        
        # tk.Button(self.root, text="View Summary", command=self.show_summary).pack()
        # tk.Button(self.root, text="Back", command=self.create_main_page).pack()
        self.show_summary()
    
    def show_summary(self):
        """Show summary of training/testing."""
        self.clear_frame()
        tk.Label(self.root, text="Summary Report").pack()
        
        summary_text = "Training and Testing completed successfully.\n\n"
        summary_text += "Parameters Used:\n"
        for key, value in self.parameters.items():
            summary_text += f"{key}: {value}\n"
        
        summary_label = tk.Label(self.root, text=summary_text, justify=tk.LEFT)
        summary_label.pack()
        
        fig, ax = plt.subplots()
        ax.bar(range(10), [random.uniform(50, 100) for _ in range(10)], color='g')
        ax.set_title("Random Summary Metrics")
        ax.set_xlabel("Metric Index")
        ax.set_ylabel("Value")
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.get_tk_widget().pack()
        canvas.draw()
        
        tk.Button(self.root, text="Back to Results", command=self.switch_to_results).pack()
    
    def clear_frame(self):
        """Clear current frame."""
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AIModelApp(root)
    root.mainloop()