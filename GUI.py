# Import required libraries
import tkinter as tk
import joblib
# Define a class for the GUI

class StockPredictorGUI:
    # The __init__ function initializes the GUI and the model
    def __init__(self):
        # Load the model
        self.model = joblib.load('model.pkl')

        # Create the main window
        self.window = tk.Tk()
        self.window.title("Stock Price Predictor")

        # Create labels and text entry fields for each feature in the model
        self.features = ['Day_of_Week', 'Month', 'Open', 'High', 'Low', 'Volume']
        self.entries = {}
        for i, feature in enumerate(self.features):
            tk.Label(self.window, text=f"Enter {feature}").grid(row=i)
            self.entries[feature] = tk.Entry(self.window)
            self.entries[feature].grid(row=i, column=1)

        # Create a button that will trigger the prediction when clicked
        tk.Button(self.window, text='Predict', command=self.make_prediction).grid(row=len(self.features))

        # Start the GUI
        self.window.mainloop()

    # The make_prediction function is called when the Predict button is clicked
    def make_prediction(self):
        # Get the user's input for each feature
        input_data = [float(self.entries[feature].get()) for feature in self.features]

        # Use the model to make a prediction
        prediction = self.model.predict([input_data])

        # Display the predicted stock price
        tk.Label(self.window, text=f"Predicted Close Price: {prediction[0]}").grid(row=len(self.features)+1)

# Create an instance of the GUI
stock_predictor = StockPredictorGUI()
