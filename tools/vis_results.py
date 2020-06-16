import numpy as np
import cv2
import copy
import argparse
import logging
import os

''' config '''
num_lmks = 5
target_shape = (64, 64)


def process_txt(txt):
    with open(txt, 'r') as f:
        lines = f.readlines()

    # each line: img_path, rects(4), lmks(10)
    info_dict_list = []    
    for line in lines:
        import pdb
        #pdb.set_trace()
        info_dict = dict()
        data = line.strip().split(' ')
        img_path = data[0]
        rects = np.array([float(x) for x in data[1:5]])
        lmks = np.array([float(x) for x in data[(5 + num_lmks * 0):(5 + num_lmks * 2)]])
        img_name = img_path.split('/')[-1]
        label = data[-2]
        #img_name = '_'.join(img_path.split('AntiSpoofingData/')[-1].split('/'))
        info_dict = {'img_path': img_path,
                    'img_name': img_name,
                    'rects': rects,
                    'lmks': lmks,
                    'label':label}
        info_dict_list.append(info_dict)

    return info_dict_list


def plot(info_dict_list, output_file):
    last_name = ''
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    for i, info_dict in enumerate(info_dict_list):
        img_name = info_dict['img_name']
        img_path = info_dict['img_path']
        rects = info_dict['rects']
        lmks = info_dict['lmks']
        label = info_dict['label']
        label = float(label)
        label = int(label)
        expression = class_names[label]
        if img_name != last_name:
            last_name = img_name
            # img = cv2.imread(img_path)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        else:
            img = img
        ref = (rects[2] - rects[0] + rects[3] - rects[1]) / 2
        cv2.rectangle(img, (int(rects[0]), int(rects[1])), (int(rects[2]), int(rects[3])), (0, 0, 255), int(ref*0.02))

        ''' vis / invis metric '''
        for j in range(num_lmks):
            if lmks[j] > 0.5:
                # green
                cv2.circle(img, (int(lmks[2 * j]), int(lmks[2 * j + 1])), int(ref*0.02), (0, 255, 0), -1)
            else:
                # yellow
                cv2.circle(img, (int(lmks[2 * j]), int(lmks[2 * j + 1])), int(ref*0.02), (0, 255, 255), -1)
            # blue
            cv2.circle(img, (int(lmks[2 * j]), int(lmks[2 * j + 1])), int(ref*0.02), (255, 128, 0), -1)
            cv2.putText(img, 'Ex:{}'.format(expression),(int(rects[0]), int(rects[1]*0.98)),color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8)

        if i+1 == len(info_dict_list):
            cv2.imwrite('./{}/{}'.format(output_file, img_name), img)
            continue
        if img_name != info_dict_list[i+1]['img_name']:
            cv2.imwrite('./{}/{}'.format(output_file, img_name), img)


def main(args):
    output_file = './Vis_expression_results'
    if os.path.exists('./{}'.format(output_file)):
        os.system('rm -r {}'.format(output_file))
    os.system('mkdir {}'.format(output_file))

    # get lmks info
    info_dict_list = process_txt(args.file)
    # plot lmks on image
    plot(info_dict_list, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot FER results')
    parser.add_argument('--file', help='lmks rects txt file path', required=True, type=str)

    args = parser.parse_args()

    main(args)