# Super_Resolution quick-start
<b>It's alwaýs a good idea to have conda installed and set this up in a new conda environment. See https://anaconda.org</b>

Install the requirements:<br>
<code>$ pip install -r requirements.txt</code>

Run the server:<br>
<code>$ uvicorn inference:app --host 0.0.0.0</code>

Open your browser and go to http://localhost:8000/form.html<br>
Select image to be super-res'ed, Click submit<br>
Profit :)

The model was created with <a href="https://perceptilabs.com">PerceptiLabs</a>, a visual frontend for TensorFlow
