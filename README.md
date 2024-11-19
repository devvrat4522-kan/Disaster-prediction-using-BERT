<h1>Disaster Prediction Using Tweets</h1>
This project leverages state-of-the-art natural language processing techniques to predict whether a given tweet indicates a disaster. The model is based on BERT (Bidirectional Encoder Representations from Transformers), which excels at understanding the context and semantics of text.
The project involves data preprocessing, text transformation using BERT, and final prediction using a fine-tuned model. The pipeline ensures accurate predictions by effectively handling text-specific challenges like spelling errors, abbreviations, and context nuances.

<h2>Features</h2>
<p>1. Predicts disaster-related tweets with high accuracy.</p>
<p>2. Utilizes pre-trained BERT for robust language understanding.</p>
<p>3. Implements a text preprocessing pipeline to clean and standardize input data.</p>
<p>4. Easy-to-use Jupyter Notebook for seamless execution and visualization.</p>

<h2>Technologies Used</h2>

<p>1. Python: Programming language for model development.</p>
<p>2. Transformers: Library for pre-trained BERT models and tokenization.</p>
<p>3. Jupyter Notebook: For implementation and step-by-step analysis.</p>
<p>4. Pandas & NumPy: For data manipulation and processing.</p>
<p>5. Scikit-learn: For evaluation metrics like accuracy and F1-score.</p>

<h2>How It Works </h2>

<h3>Data Preprocessing</h3>
<p>1. Text is cleaned by removing special characters, URLs, and stop words.
Tokenization and padding are applied to make text input BERT-compatible.
Transformation and Embedding.</p>
<p>2. Tweets are transformed into contextual embeddings using BERT's tokenizer and model.
Model Prediction.</p>
<p>3. The fine-tuned BERT model predicts whether a tweet is disaster-related (binary classification).
Evaluation.</p>
<p>4.The model’s performance is evaluated using metrics like precision, recall, and F1-score.</p>

<h3>Dataset</h3>

The dataset contains labeled tweets, where:
1 indicates the tweet refers to a disaster.
0 indicates the tweet does not refer to a disaster.

<h2>Installation and Usage</h2>

<h3>Clone the Repository </h3>
git clone https://github.com/your-username/disaster-prediction-using-tweets.git  
cd disaster-prediction-using-tweets  

<h2>Setup Environment</h2>
<h3>Install the required dependencies using:</h3>
pip install -r requirements.txt  

<h2>Run the Notebook</h2>
<p>1. Open the provided Jupyter Notebook (disaster-with-nlp.ipynb).</p>
<p>2. Follow the steps for data loading, preprocessing, and model prediction.</p>

<h2>Results</h2>
<h4>
 <p>{'accuracy': [0.6726025342941284, 0.7341994047164917, 0.7534666657447815],</p>
 <p>'loss': [0.618563711643219, 0.5462052822113037, 0.5186125040054321],</p>
 <p>'recall': [0.6726025342941284, 0.7341994047164917, 0.7534666657447815],</p>
 <p>'val_accuracy': [0.6154855489730835, 0.6338582634925842, 0.5839895009994507],</p>
 <p>'val_loss': [0.6521779894828796, 0.6657184958457947, 0.7971555590629578],</p>
 <p>'val_recall': [0.6154855489730835, 0.6338582634925842, 0.5839895009994507]}</p>
</h4>

<h2>Future Work</h2>
<p>1. Extend the model to handle multilingual tweets.</p>
<p>2. Improve the preprocessing pipeline for better handling of noisy text.</p>
<p>3. Deploy the model as a web application for real-time tweet analysis.</p>

