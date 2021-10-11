import tensorflow as tf
from PIL import Image
import numpy as np
#import argparse
from skimage.util import view_as_windows
import io
from fastapi import FastAPI, Response, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
import webbrowser
import tensorflow_addons as tfa
from starlette.responses import FileResponse
import os

app = FastAPI()



#parser = argparse.ArgumentParser()
#parser.add_argument('--input', type=str, help='Input file', required=True)
#parser.add_argument('--output', type=str, help='Output file', required=True)
#args = parser.parse_args()

# Load the model
model = tf.keras.models.load_model('./model/')

# Load the image and convert it to RGB (in case of grayscale, RGBA etc..)
#img = Image.open(args.input).convert('RGB')
@app.post("/predict/")
async def predict(bild: UploadFile = File(...), magnify: int = Form(...)):
    file_name = "{0}_{2}x.png".format(*os.path.splitext(bild.filename), magnify)
    img = Image.open(bild.file).convert("RGB")
    #img_1 = np.asarray(img)
    #img_1 = tfa.image.gaussian_filter2d(img_1, filter_shape=3, sigma=1.0)
    #img = tf.keras.preprocessing.image.array_to_img(img_1)
    W, H = img.size
    
    # Upsample imgage 4x (it's because the way this model was trained)
    W = W * magnify
    H = H * magnify
    img = img.resize((W, H))
    
    # Since the model only takes the image in blocks of 208x208, we need to pad the image to be evenly divideable with this size
    #W_1 = np.ceil(W / 208) * 208
    #H_1 = np.ceil(H / 208) * 208
    #padded_img = Image.new('RGB', (np.uint16(W_1), np.uint16(H_1)), 0)
    #padded_img.paste(img)
    
    # Convert the padded image to an array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.pad(img_array, ((0,224), (0,224), (0, 0)))
    
    # Now slice it to 208x208 blocks
    img_array = view_as_windows(img_array, (224, 224, 3), step=208)
    
    # Pad each block with edge pixels to 224x224 (the actual input size for the model)
    x = img_array.shape
    
    
    # Get the number of blocks for the prediction loop
    block_rows, block_cols = x[:2]
     
    
    # Create an empty array where we put the predicted blocks
    img_array_out = np.zeros((block_rows * 208, block_cols * 208, 3), dtype = np.float32)
    
    for b in range(block_rows):
        for a in range(block_cols):
            pred_in = img_array[b,a,:,:].reshape(1, 224, 224, 3) # pick out the block to be predicted
            predict = np.asarray(model(pred_in)['target']).reshape(224, 224, 3) # make the prediction and reshape output (omit batch)
            #predict = tfa.image.gaussian_filter2d(predict, filter_shape=12, sigma=4.0)
            img_array_out[(b * 208):(b * 208) + 208, (a * 208):(a * 208) + 208, :] = predict[8:216, 8:216, :] # put the predicted block in it's right place in the new array
    img_out = tf.keras.preprocessing.image.array_to_img(img_array_out) # convert array to image
    img_out = img_out.crop((0, 0, W, H)) # crop out the padding
    img_out.save("4x.png", format='png')
    #img_out.show() # show it
    #img_byte_array = io.BytesIO()
    #img_out.save(img_byte_array, format='png')
    #img_byte_array = img_byte_array.getvalue()
    return FileResponse("4x.png", media_type="application/octet-stream", filename=file_name)
    #img_out.save(args.output) # save it
    
app.mount("/", StaticFiles(directory="form"), name="form")

webbrowser.open("http://localhost:8000/form.html")
