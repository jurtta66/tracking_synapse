import cv2 as cv
import sort
import torch
import numpy as np
import os


def path(folder_path):
    images_path = []
    for filename in sorted(os.listdir(folder_path)):
        images_path.append(folder_path + '/' + filename)
    return images_path


def POINTS(event, x, y, flags, param):
    if event == cv.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)


def main(video_path):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    tracker = sort.Sort()
    trajectory = dict() #Здесь будет траектория каждого человечка


    cv.namedWindow('FRAME')
    cv.setMouseCallback('FRAME', POINTS)
    frames = path(video_path)


    for frame in frames:
        image = cv.imread(frame)
        image = cv.resize(image,(1020,500))


        results = model(image)

        coord = []
        for _, row, in results.pandas().xyxy[0].iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            b = str(row['name'])
            if 'person' in b:
                coord.append([x1, y1, x2, y2])


        boxes_ids = tracker.update(np.array(coord))
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            x, y, w, h, id = int(x), int(y), int(w), int(h), int(id)
            if id in trajectory:
                trajectory[id].append([x, y])
            else:
                trajectory[id] = []
                trajectory[id].append([x, y])


            cv.rectangle(image, (x, y), (w, h), (255, 0, 255), 2)
            cv.putText(image, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv.imshow('Image',image)


        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
    return trajectory


if __name__ == '__main__':
    video_path = '/home/mikhail/work/CompVision/opencvtutorial_env/MOT15/test/ETH-Crossing/img1'
    main(video_path)
