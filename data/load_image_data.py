import numpy as np
import os
from PIL import Image
from itertools import product
from datetime import datetime

def load_images(image_base_dir):

    locations = set()
    data = {'timestamp': [], 'block': [], 'district': [], 'image': []}

    for subdir, _, files in os.walk(image_base_dir):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".tif"):

                # Extract location and date information from the file path
                info = [x.lower() for x in filepath.split('/')[-3:]]
                locations.add((info[0], info[1]))

                # Extract date information from the file_name
                date = info[2][:8]
                date_object = datetime.strptime(date, '%Y%m%d')
                timestamp = date_object.strftime('%Y-%m-%d %H:%M:%S')
                # print (timestamp, info)

                # Add all these parameters to the dataset
                data['district'].append(info[0])
                data['block'].append(info[1])
                data['timestamp'].append(timestamp)

                img = Image.open(filepath)
                img_array = np.array(img)
                data['image'].append(img_array)
            
    return data