# -*- coding: utf-8 -*-
"""
author: SUVANKAR BANERJEE
email: suvankarbanerjee73@gmail.comm
license: MIT License
year: 2024
location: Natural Science Laboratory, Dept of Pharm. tech, JU, INDIA.
"""

## Dependencies ##
import platform
import threading
import os
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image
import matplotlib

try:
    matplotlib.use('TkAgg')  # Set the backend
    import matplotlib.pyplot as plt
except ImportError:
    print('Error importing matplotlib')

try:
    from pylda import LDApy
except ImportError:
    print('Error importing pylda')

import platform

class App:
    __version__ = 0.23
    def __init__(self, master):
        # main
        self.LDA = LDApy()

        # Input variables
        self.train_file_name = ""
        self.test_file_name = ""
        self.output_file_name = ""
        self.tolerance = ""
        self.tol_var = tk.StringVar(value="0.0001")
        self.y_name= ""
        self.index= ""
        
        self.master = master
        self.master.geometry(self.get_tkinter_geometry(50))  # Use the function here
        self.master.resizable(1, 1)
        self.master.title("LinDy v0.2")
        
        # Set window icon        
        if platform.system() == "Linux":
        	try:
        		logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon', 'logo.xbm')
        		self.master.iconbitmap('@' + logo_path)
        	except Exception as e:
        		print(f"Failed to set icon: {e}")
        else:
            try:
            	logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon', 'logo.ico')
            	self.master.iconbitmap(logo_path)
            except Exception as e:
            	print(f"Failed to set icon: {e}")
            
        # Configure grid layout
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_rowconfigure(0, weight=1)

        # Primary i/o frames
        self.main_frame = ttk.Frame(self.master, borderwidth=1)
        self.main_frame.pack(fill="x", expand=True)      
        
        # Call sidebar function
        self.main_pyLDA_sidebar(self.main_frame)
        
        # Additional frame for text box
        self.text_frame = ttk.Frame(self.master, borderwidth=1)
        self.text_frame.pack(fill="both", expand=True)
                
        # Add a text box to display the output text data in tab 2
        self.text_box = tk.Text(self.text_frame, height=8, width=45, font=("Arial", 9))
        self.text_box.pack(fill="x", expand=True, padx=5, pady=5)              

    # Main pyLDA sidebar function
    def main_pyLDA_sidebar(self, frame):
        self.frame = frame
        
        # Input fields
        self.train_entry = ttk.Entry(self.frame, width=25)
        self.train_entry.grid(row=1, column=0, padx=10, pady=10)
        self.test_entry = ttk.Entry(self.frame, width=25)
        self.test_entry.grid(row=2, column=0, padx=10, pady=10)

        # Buttons for tab 1
        self.train_button = ttk.Button(self.frame, text="Load Training Data", width=20, command=self.browse_train_file)
        self.train_button.grid(row=1, column=1, padx=10, pady=10)
        self.test_button = ttk.Button(self.frame, text="Load Test Data", width=20, command=self.browse_test_file)
        self.test_button.grid(row=2, column=1, padx=10, pady=10)

        # Button to enter y_column
        self.ycol_button = ttk.Button(self.frame, text="Enter Dependent Data", width=25, command=self.dialog_ycol)
        self.ycol_button.grid(row=3, column=0, padx=10, pady=10)

        # Button to enter index_column
        self.indexcol_button = ttk.Button(self.frame, text="Enter Index Column", width=20, command=self.dialog_indexcol)
        self.indexcol_button.grid(row=3, column=1, padx=10, pady=10)

        # OptionMenu for selecting solver
        self.optionmenu_label = ttk.Label(self.frame, text="Select Solver:")
        self.optionmenu_label.grid(row=4, column=0, padx=10, pady=5, sticky="nw")
        self.optionmenu_solver = ttk.Combobox(self.frame, values=["svd", "eigen", "lsqr"])
        self.optionmenu_solver.set("svd")  # set initial value
        self.optionmenu_solver.grid(row=5, column=0, padx=10, pady=0, sticky="nw")

        # Labels and entry for tolerance
        self.tol_label = ttk.Label(self.frame, text="Enter Tolerance:")
        self.tol_label.grid(row=4, column=1, padx=10, pady=5, sticky="nw")
        self.tol_entry = ttk.Entry(self.frame, textvariable=self.tol_var)
        self.tol_entry.grid(row=5, column=1, padx=10, pady=0, sticky="nw")
        
        # Use trace on the StringVar
        self.tol_var.trace("w", lambda name, index, mode, sv=self.tol_var: self.tolerance_callback())

        # Button to run LDA
        self.run_button = ttk.Button(self.frame, text="Run LDA", width=40, command=self.run_lda)
        self.run_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10)
  
        # Button to save summary
        self.save_summary_button = ttk.Button(self.frame, text="Save Summary", width=40, command=self.save_pylda_summary)
        self.save_summary_button.grid(row=7, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        
        # Add buttons for displaying plots
        self.roc_button = ttk.Button(self.frame, text="ROC Curve", width=20, command=self.show_roc_curve)
        self.roc_button.grid(row=8, column=0, padx=5, pady=2, sticky="nsew")
        self.confusion_button = ttk.Button(self.frame, text="Confusion Matrix", width=20, command=self.show_confusion_matrix)
        self.confusion_button.grid(row=8, column=1, padx=5, pady=2, sticky="nsew")

    def get_tkinter_geometry(self, percent_of_screen, xpad=None, ypad=None):
        '''
        Given percent of monitor size in floating point eg: 10 % = 10.0, calculates
        tkinter geometry for each monitor attached to computer
 
        returns: list holding tkinter geometry strings padded with xpad and ypad
                or centered if xpad is None.
                 
                None if bad pct passed
        '''
 
        if not isinstance(percent_of_screen, float):
            print("requires float percent eg: 10.0 for 10%")
            return
 
        pct = percent_of_screen / 100
 
        # Get screen size using Tkinter's built-in methods
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        cwidth = int(screen_width * pct)
        cheight = int(screen_height * pct)
 
        xoff = xpad
        yoff = ypad
        if xpad is None:
            xoff = int((screen_width - cwidth) / 2)
            yoff = int((screen_height - cheight) / 2)
        return f"{cwidth}x{cheight}+{xoff}+{yoff}"
    
    ## App functions
    def dialog_ycol(self):
        """
        Open a dialog box to enter the dependent column name.
        """
        self.y_name = simpledialog.askstring("Enter Dependent Column", "Please enter the dependent column name:")
        
    def dialog_indexcol(self):
        """
        Open a dialog box to enter the index column name.
        """
        self.index = simpledialog.askstring("Enter Index Column", "Please enter the index column name:")

    
    def browse_train_file(self):
        try:
            self.train_file_name = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"),
                                                                         ('text files', '*.txt'),
                                                                         ('All files', '*.*')])
            self.train_entry.delete(0, tk.END)
            self.train_entry.insert(0, self.train_file_name)
        except FileNotFoundError:
            print("User canceled the file dialog.")

    def browse_test_file(self):
        try:
            self.test_file_name = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"),
                                                                        ('text files', '*.txt'),
                                                                        ('All files', '*.*')])
            self.test_entry.delete(0, tk.END)
            self.test_entry.insert(0, self.test_file_name)
        except FileNotFoundError:
            print("User canceled the file dialog.")

    # Consider further actions on dropdown choice if required
    def tolerance_callback(self):
        selected_tolerance = self.tol_var.get()
        try:
            selected_tolerance = float(selected_tolerance)
            print("Tolerance input:", selected_tolerance)
            return selected_tolerance
        except ValueError:
            print("Invalid tolerance input")
            return None

    # Consider further actions on dropdown choice if required
    def optionmenu_callback(self, choice):
        print("Optionmenu dropdown clicked:", choice)

    # Additional method to validate input fields
    def validate_input_fields(self):
        if not (self.train_file_name and self.test_file_name):
            messagebox.showerror("Error", "Please select all input files and output path.")
            return False
        return True

    # Additional method to handle completion message
    def success_msg(self):
        messagebox.showinfo("Status", "Job completed")

    # Define a function to clear the Entry Widget Content
    def green_slate(self):
        # clear gui input fields
        self.train_entry.delete(0, 'end')
        self.test_entry.delete(0, 'end')

    # Function to write to the text box
    def write_to_text_box(self, content):
        # If a canvas is available, draw on it
        if hasattr(self, 'text_box') and self.text_box:
            # If canvas is not available, write to the text box
            self.text_box.insert(tk.END, content)
            print("Checkpoint 6")
        else:
            print("Error")
            
    # Method to handle saving both CSV and text content
    def save_pylda_summary(self):
        # Check if LDA object is properly initialized
        if not hasattr(self, 'LDA') or not hasattr(self.LDA, 'X_train'):
            messagebox.showwarning("Warning", "Please run LDA analysis first.")
            return
        
        # Validate that LDA analysis has been run and there is some text content
        if hasattr(self, 'text_box') and self.text_box:
            # Check if X_train attribute exists in the LDA object
            if not hasattr(self.LDA, 'X_train'):
                messagebox.showwarning("Warning", "Please run LDA analysis first.")
                return
            
            try:
                # get output file
                self.output_file_name = filedialog.asksaveasfilename(title="Output File Name",
                                                                      filetypes=[("dat Files", "*.dat"),
                                                                                 ('All files', '*.*')])
            except:
                messagebox.showerror("Error", "Please provide output file name.")
            
            # Save CSV file
            self.LDA.save_predictions_to_csv(self.output_file_name)
            
            # Save text content as a .dat file with the prefix "Summary_Performance_"
            summary_file_name = f"Summary_Performance_{os.path.basename(self.output_file_name)}.dat"
            summary_file_path = os.path.join(os.path.dirname(self.output_file_name), summary_file_name)
            with open(summary_file_path, 'w') as summary_file:
                summary_file.write(self.text_box.get("1.0", tk.END))
                
            # Show success message
            messagebox.showinfo("Status", "Summary and CSV files saved successfully.")
        else:
            messagebox.showwarning("Warning", "No summary to save. Please run LDA analysis first.")
                
    def show_roc_curve(self):
        self.LDA.plot_roc_curve()
        plt.show()

    def show_confusion_matrix(self):
        self.LDA.plot_confusion_matrix()
        plt.show(block=False)
        plt.pause(0.01)

    def run_lda(self):
        # Clear the text box before inserting new content
        self.text_box.delete(1.0, "end")
        print("Checkpoint 0")

        self.train_file_name = self.train_entry.get()
        self.test_file_name = self.test_entry.get()
        self.selected_solver = self.optionmenu_solver.get()  # Get the selected solver from the combobox
        self.selected_tolerance = float(self.tolerance_callback())

        print("Tolerance = ", self.selected_tolerance)
        print("Solver = ", self.selected_solver)

        # Validate input fields
        if not self.validate_input_fields():
            return

        # Create and run the LDA analysis
        lda_obj = self.LDA
        lda_obj.load_data(self.train_file_name, self.test_file_name, self.y_name, self.index)  # Load data first

        try:
            lda_obj.fit_lda(self.selected_solver, self.selected_tolerance)
            lda_obj.predict_lda()
            lda_obj.calculate_metrics()

            # Display success message
            self.success_msg()

            print("Checkpoint 3")
            # Updated code to output metrics
            self.output_metrics = lda_obj.output_metrics_string()

            # Insert the formatted string into the text box
            self.write_to_text_box(self.output_metrics)
            print("Checkpoint 4")

            # Clear entries
            self.green_slate()
            print("Checkpoint 5")

        except Exception as e:
            traceback.print_exc()
            messagebox.showinfo("Error", f"An error occurred: {str(e)}")
    

if __name__ == "__main__":    
    root = tk.Tk()
    app = App(root)
    root.mainloop()


