from flask import Flask, render_template, redirect, jsonify, request,url_for  
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
from PIL import Image,ImageDraw
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)


upload_folder = os.path.join('static/upload', 'test')
app.config['UPLOAD'] = upload_folder  


adm=tf.keras.models.load_model('static/model/AMD.h5')      
dr=tf.keras.models.load_model('static/model/DR.h5')
glancoma=tf.keras.models.load_model('static/model/glancoma.h5')
catoract=tf.keras.models.load_model('static/model/catoract.h5')
hyper=tf.keras.models.load_model('static/model/Hyper.h5')
myopia=tf.keras.models.load_model('static/model/myopia.h5')
    
def crop(img):
    h,w=img.size
    mask=Image.new('L', img.size ,0)
    draw=ImageDraw.Draw(mask)
    draw.pieslice([(h//13,w//13),(h-h//13,w-w//13)], 0, 360,fill=255) 
    mask1=Image.new('RGB', img.size,(0,0,0))
    crop=Image.composite(img,mask1, mask)
    return crop
    
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/index.html")
def upload():
    return render_template("index.html")


@app.route("/index.html",methods=['POST'])
def process():  
    try:
        if request.method == 'POST':
            file = request.files['image']
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD'], "sat.jpeg"))
            img_path = os.path.join(app.config['UPLOAD'], "sat.jpeg")
    except:        
        err="Try Again"
        return render_template('index.html',err=err)     
    # img=Image.open(img_path)
    # img=crop(img) 
    # img.save(img_path)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_image = test_datagen.flow_from_directory('D:/CODE/eye_project/static/upload',target_size = (128,128),batch_size = 32)

    test_image1 = test_datagen.flow_from_directory('C:/Users/HP/OneDrive/Desktop/archive (2)/all/val/normal',target_size = (128,128),batch_size = 32)
    res1 = adm.predict(test_image1)  
    res2 = dr.predict(test_image1)  
    res3 = glancoma.predict(test_image1)  
    res4 = catoract.predict(test_image1)  
    res5 = hyper.predict(test_image1)  
    res6 = myopia.predict(test_image1)  

 
    for i in range(len(res1)):
        k=[]
        k.append(np.argmax(res1[i]))
        k.append(np.argmax(res2[i]))
        k.append(np.argmax(res3[i]))
        k.append(np.argmax(res4[i]))
        k.append(np.argmax(res5[i]))
        k.append(np.argmax(res6[i]))
        a=[1,0,1,0,0,0]
        if k is a:
            print(i)



    res_adm = int(np.argmax(adm.predict(test_image)))
    res_dr = int(np.argmax(dr.predict(test_image)))
    res_glancoma = int(np.argmax(glancoma.predict(test_image)))
    res_catoract = int(np.argmax(catoract.predict(test_image)))
    res_hyper = int(np.argmax(hyper.predict(test_image)))
    res_myopia = int(np.argmax(myopia.predict(test_image)))

    adm_dic={0:"Dry_ADM",1:"Normal",2:"Wet_AMD"}
    dr_dic={0:"Normal",1:"Mild_DR",2:"Modrate_DR",3:"Severe_DR",4:"Proliferate_DR"}
    glancoma_dic={1:"Normal",0:"Glaucoma"}
    catoract_dic={0:"Normal",1:"Catarat"}
    hyper_dic={0:"Normal",1:"Hypertensive"}
    myopia_dic={0:"Normal",1:"Pathological_Myopia"}
    result_dic={"Dry_AMD":"No","Wet_AMD":"No","Mild_DR":"No","Modrate_DR":"No","Severe_DR":"No","Proliferate_DR":"No",
    "Glaucoma":"No","Cataract":"No","Pathological_Myopia":"No","Hypertensive":"No"}  
    all_list=(adm_dic[res_adm],dr_dic[res_dr],catoract_dic[res_catoract],glancoma_dic[res_glancoma],hyper_dic[res_hyper],myopia_dic[res_myopia])
    for i in result_dic:
        if i in all_list:
            result_dic[i]="Yes"      

    if len(list(set(all_list)))==1:
        result_dic["Normal"]="Yes"  
    else:
        result_dic["Normal"]="No"  


    return render_template("report.html",result_dic=result_dic,img=img_path)    


if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000,debug=True)    
