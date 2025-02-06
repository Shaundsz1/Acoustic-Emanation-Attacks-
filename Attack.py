import os
import sys
import numpy as np
from scipy.fft import fft
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Adjust path to the correct handout directory
handout_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'handout')
sys.path.insert(0, handout_dir)

try:
    from extractKeyStroke import extractKeyStroke
except ModuleNotFoundError:
    print(f"Could not find 'extractKeyStroke.py' in {handout_dir}")
    sys.exit(1)

# Function to collect and preprocess training data
def collect_training_data(file_list, data_dir):
    training_data = []
    with open("thresholds_log.txt", "w") as thresh_log:
        for filename in file_list:
            file_path = os.path.join(data_dir, filename)
            print(f"Processing: {file_path}")
            thresh = 17.0  # Initial threshold value

            # Adjust threshold to obtain exactly 100 peaks
            while True:
                peaks, click_count, keystroke_info = extractKeyStroke(file_path, 100, thresh)
                if len(peaks) == 100:
                    thresh_log.write(f"{filename}: {thresh}\n")
                    break
                thresh -= 0.5

            # Standardize keystroke length
            desired_length = 2048  
            for stroke in peaks:
                standardized_stroke = adjust_length(stroke, desired_length)
                fft_transformed = np.abs(fft(standardized_stroke))
                training_data.append(fft_transformed)
    return np.array(training_data)

# Function to standardize signal length by padding or truncating
def adjust_length(signal, target_length):
    if len(signal) > target_length:
        return signal[:target_length]
    else:
        return np.pad(signal, (0, target_length - len(signal)), 'constant')

# Function to process input files test or secret and apply FFT after adjusting threshold
def process_input_file(input_file, num_strokes=8):
    thresh = 17.0
    while True:
        strokes, _, _ = extractKeyStroke(input_file, num_strokes, thresh)
        if len(strokes) == num_strokes:
            break
        thresh -= 0.5

    # Standardize keystroke length
    processed_data = []
    desired_length = 2048  
    for stroke in strokes:
        standardized_stroke = adjust_length(stroke, desired_length)
        fft_transformed = np.abs(fft(standardized_stroke))
        processed_data.append(fft_transformed)
    return np.array(processed_data)

# Function to assess model performance on test data
def assess_model_performance(model, scaler, data_dir, test_files):
    index = 0
    for test_filename in test_files:
        full_path = os.path.join(data_dir, test_filename)
        test_inputs = process_input_file(full_path, num_strokes=8)
        test_inputs = scaler.transform(test_inputs)
        probability_predictions = model.predict_proba(test_inputs)

        top1, top2, top3 = 0, 0, 0
        actual_class = model.classes_[index]

        for probs in probability_predictions:
            ranked_indices = np.argsort(probs)[::-1]
            if model.classes_[ranked_indices[0]] == actual_class:
                top1 += 1
            elif model.classes_[ranked_indices[1]] == actual_class:
                top2 += 1
            elif model.classes_[ranked_indices[2]] == actual_class:
                top3 += 1

        print(f"Testing soundfile {test_filename}: First: {top1} | Second: {top2} | Third: {top3}")
        index += 1

# Function to reveal secrets by displaying top predictions
def decode_secret(model, scaler, secret_inputs):
    secret_inputs = scaler.transform(secret_inputs)
    probability_predictions = model.predict_proba(secret_inputs)

    first_choices, second_choices, third_choices = "", "", ""
    for probs in probability_predictions:
        ranked_indices = np.argsort(probs)[::-1]
        first_choices += model.classes_[ranked_indices[0]]
        second_choices += model.classes_[ranked_indices[1]]
        third_choices += model.classes_[ranked_indices[2]]

    print("Secret Predictions:")
    print(f"First Most likely:{first_choices}")
    print(f"Second likely:{second_choices}")
    print(f"Third likely:{third_choices}")

# Function to save the trained model and scaler
def save_trained_model(model, scaler):
    with open('trained_keyboard_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

# Function to load the saved model and scaler
def load_trained_model_and_scaler():
    with open('trained_keyboard_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

def execute_training():
    data_dir = os.path.join(os.path.dirname(__file__), 'Lab4_Python', 'data', 'data')
    train_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.wav') and 'test' not in f and 'secret' not in f])

    # Collect training data and generate labels
    train_inputs = collect_training_data(train_files, data_dir)
    labels = [chr(97 + i) for i in range(26) for _ in range(100)]  # 100 samples per character

    # Normalize the training data
    scaler = StandardScaler()
    train_inputs = scaler.fit_transform(train_inputs)

    # Create and train the neural network model
    mlp_model = MLPClassifier(hidden_layer_sizes=(150,), max_iter=3000, verbose=True)
    mlp_model.fit(train_inputs, labels)

    # Save the model and scaler
    save_trained_model(mlp_model, scaler)

# Function to test the model and recover secrets
def execute_testing_and_secret_recovery():
    model, scaler = load_trained_model_and_scaler()
    data_dir = os.path.join(os.path.dirname(__file__), 'Lab4_Python', 'data', 'data')

    # Identify test and secret files
    all_filenames = sorted(os.listdir(data_dir))
    test_filenames = [f for f in all_filenames if 'test' in f]
    secret_filenames = [f for f in all_filenames if 'secret' in f]

    # Evaluate the model on test files
    print("Testing Model Accuracy on Test Files:")
    assess_model_performance(model, scaler, data_dir, test_filenames)

    # Decode secrets from secret files
    print("\nRecovering Secret Data:")
    for secret_filename in secret_filenames:
        secret_path = os.path.join(data_dir, secret_filename)
        # Assume each secret file has 8 keystrokes
        secret_inputs = process_input_file(secret_path, num_strokes=8)
        print(f"\nSecret File: {secret_filename}")
        decode_secret(model, scaler, secret_inputs)

# Main execution flow
if __name__ == "__main__":
    execute_training()
    execute_testing_and_secret_recovery()
