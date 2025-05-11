from flask import Flask, render_template, request
import numpy as np
import torch
import pickle
import re
import os 
from tape import UniRepModel, TAPETokenizer
from tensorflow.keras.models import model_from_json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')

# Set device for Torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

# -----------------------------
# Load Models and Resources
# -----------------------------
with open(os.path.join(MODEL_DIR, 'Multiheaded_unirep.json'), 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights(os.path.join(MODEL_DIR, 'Multiheaded_unirep.weights.h5'))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)


unirep_model = UniRepModel.from_pretrained('babbler-1900')
unirep_model = unirep_model.to(DEVICE)
tokenizer = TAPETokenizer(vocab='unirep')

# -----------------------------
# Utility Functions
# -----------------------------
def is_valid_protein_sequence(sequence):
    """Check if the sequence contains only valid amino acid letters"""
    return bool(re.fullmatch(r'^[ACDEFGHIKLMNPQRSTVWY]+$', sequence, re.IGNORECASE))

def parse_fasta(fasta_text):
    """Parse FASTA text with enhanced error checking"""
    lines = [line.strip() for line in fasta_text.splitlines() if line.strip()]
    
    if not lines:
        raise ValueError("Empty input")
    
    # Check for multiple sequences
    if sum(1 for line in lines if line.startswith(">")) > 1:
        raise ValueError("Multiple protein sequences detected. Please input only one protein sequence at a time.")
    
    protein_id = "Unknown"
    sequence = []
    
    for line in lines:
        if line.startswith(">"):
            protein_id = line[1:].split()[0] if len(line) > 1 else "Unknown"
        else:
            sequence.append(line.upper())
    
    full_sequence = ''.join(sequence)
    
    if not full_sequence:
        raise ValueError("No sequence found in the FASTA input")
    
    if not is_valid_protein_sequence(full_sequence):
        raise ValueError("Invalid protein sequence - contains non-amino acid characters")
    
    return protein_id, full_sequence

def get_unirep_embedding(sequence):
    """Generate UniRep embedding with error handling"""
    try:
        with torch.no_grad():
            token_ids = torch.tensor([tokenizer.encode(sequence)]).to(DEVICE)
            output = unirep_model(token_ids)
            unirep_output = output[0].mean(dim=1)
            unirep_output = unirep_output[:, :1024]
            features = unirep_output.cpu().numpy()[0]
        return features
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding: {str(e)}")

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", page='home')

@app.route("/model", methods=["GET", "POST"])
def model_page():
    protein_id = None
    label = None
    error = None
    
    if request.method == "POST":
        fasta_text = request.form.get("fasta", "").strip()
        
        try:
            if not fasta_text:
                raise ValueError("No FASTA input provided")
            
            # Check if the input starts with '>'
            if not fasta_text.startswith(">"):
                raise ValueError("Invalid input: FASTA sequence must start with '>'")
            
            protein_id, sequence = parse_fasta(fasta_text)
            
            if len(sequence) > 1000:  # Example length check
                raise ValueError("Sequence too long (max 1000 amino acids)")
                
            features = get_unirep_embedding(sequence)
            features = features.reshape(1, -1)
            features_scaled = scaler.transform(features)
            num_features = features_scaled.shape[1]
            features_reshaped = features_scaled.reshape(1, 1, num_features, 1)

            pred_prob = model.predict(features_reshaped)[0][0]
            label = "Antigenic" if pred_prob > 0.5 else "Non-antigenic"
            
        except ValueError as ve:
            error = f"Input Error: {str(ve)}"
        except RuntimeError as re:
            error = f"Processing Error: {str(re)}"
        except Exception as e:
            error = f"Unexpected Error: {str(e)}"

    return render_template("index.html",
                         page='model',
                         protein_id=protein_id,
                         label=label,
                         error=error)


@app.route("/about")
def about():
    return render_template("index.html", page='about')

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0', port=10000)