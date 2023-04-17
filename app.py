########################################################
## This file contains the code for the Flask web app  ##
########################################################

import torch
from utils.model_utils import load_model
from utils.data_utils import return_kmer, is_dna_sequence
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="templates")

model_config = {
    "model_path": "results/classification/model", # Path to the trained model
    "num_classes": 6,
}

model, tokenizer, device = load_model(model_config, return_model=True)

# Dictionary to convert the predicted class by the model to the class name
class_names_dic = {
    1: "SARS-COV-1",
    2: "MERS",
    3: "SARS-COV-2",
    4: "Ebola ",
    5: "Dengue",
    6: "Influenza",
}

KMER = 3
SEQ_MAX_LEN = 512

def huggingface_predict(input):
    """
    The input is passed to this function and the model makes a prediction
    
    Parameters
    ----------
    input : str
        The input sequence to be classified

    Returns
    -------
    predicted_class : int
        The predicted class of the input sequence
    """
    
    # Check if the input sequence is a DNA sequence
    if not is_dna_sequence(input):
        return "Invalid Input. Please enter your sequence in upper case", 0

    kmer_seq = return_kmer(input, K=KMER)

    # Tokenize the input sequence
    inputs = tokenizer(kmer_seq, padding=True, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)

    # Pass the tokenized inputs through the model to make a prediction
    outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits).item() + 1 # add 1 to convert from 0-indexed to 1-indexed classes
    prediction_probs = torch.softmax(outputs.logits, dim=1).tolist()[0]

    prediction_probability = prediction_probs[predicted_class - 1]
    prediction_probability = round(prediction_probability, 3) * 100

    # Convert the predicted class to the class name
    predicted_class = class_names_dic[predicted_class]
    
    return predicted_class, prediction_probability

@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/predict', methods=['POST']) # handle the post request from the form in index.html
def predict():
    input = request.form['input_sequence']
    prediction, probability = huggingface_predict(input)
    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)