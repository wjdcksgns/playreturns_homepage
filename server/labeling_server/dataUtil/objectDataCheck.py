import pandas as pd
import imagehash as ih
import cv2

from PIL import Image

def object_check():
    
    object_data = pd.read_csv('./dataUtil/object_data.csv')['DataNames'].tolist()
    
    return object_data


def object_insert(name, num):
    
    df = pd.read_csv('./dataUtil/object_data.csv')

    df.loc[-1] = [name, num]

    df.to_csv('./dataUtil/object_data.csv',index = False) 

    object_data = pd.read_csv('./dataUtil/object_data.csv')['DataNames'].tolist()

    print(object_data)
    
    del df
    
    return object_data


def image_hash(origin, const):
    
    hash_type = ['average', 'perceptive', 'wavelet']

    origin = cv2.cvtColor(origin, cv2.COLOR_RGB2BGR)
    origin_img = Image.fromarray(origin)

    const = cv2.cvtColor(const, cv2.COLOR_RGB2BGR)
    const_img = Image.fromarray(const)

    results = []

    for h_type in hash_type:

        try:

            if h_type == 'average':

                origin_hash = ih.average_hash(origin_img)
                const_hash = ih.average_hash(const_img)

                results.append((origin_hash - const_hash) / len(origin_hash.hash) ** 2)
            
            elif h_type == 'perceptive':

                origin_hash = ih.phash(origin_img)
                const_hash = ih.phash(const_img)

                results.append((origin_hash - const_hash) / len(origin_hash.hash) ** 2)

            elif h_type == 'wavelet':

                origin_hash = ih.whash(origin_img)
                const_hash = ih.whash(const_img)

                results.append((origin_hash - const_hash) / len(origin_hash.hash) ** 2)
        
        except Exception as e:
            
            print(e)

            pass

    return results