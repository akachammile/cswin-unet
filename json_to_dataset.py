import numpy as np
import PIL.Image
import json
import os
import os.path as osp
from labelme import utils
import base64

if __name__ == '__main__':
    jpgs_path = "datasets/JPEGImages"
    pngs_path = "datasets/SegmentationClass"
    classes = ["_background_", "leaf", "powdery"]

    count = os.listdir("./datasets/before/")
    for i in range(0, len(count)):
        path = os.path.join("./datasets/before", count[i])

        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))

            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value

            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
                print(label_names)
            assert label_values == list(range(len(label_values)))

            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            PIL.Image.fromarray(img).save(osp.join(jpgs_path, count[i].split(".")[0] + '.jpg'))

            new = np.zeros([np.shape(img)[0], np.shape(img)[1]], dtype=np.uint8)
            for name in label_names:
                index_json = label_names.index(name)
                print(name)
                index_all = classes.index(name)
                new = new + index_all * (np.array(lbl) == index_json)

            # 直接使用 PIL 保存
            PIL.Image.fromarray(new).save(osp.join(pngs_path, count[i].split(".")[0] + '.png'))
            print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')
