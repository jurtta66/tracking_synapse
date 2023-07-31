import cv2 as cv
import sort
import torch
import numpy as np
import os
from skimage.transform import resize


video_path = '/home/mikhail/work/CompVision/opencvtutorial_env/MOT15/train/ADL-Rundle-6/img1'
frame_width = 1280
frame_height = 720
cell_size = 80 # 40x40 pixel
n_cols = frame_width // cell_size
n_rows = frame_height // cell_size
alpha = 0.4
heat_matrix = np.zeros((n_rows, n_cols))


def get_row_col(x, y):
    row = y // cell_size
    col = x  // cell_size
    return row, col


def draw_grid(image):
    for i in range(n_rows):
        start_point = (0, (i + 1) * cell_size)
        end_point = (frame_width, (i + 1) * cell_size)
        color = (255, 255, 255)
        thickness = 1
        image = cv.line(image, start_point, end_point, color, thickness)

    for i in range(n_cols):
        start_point = ((i + 1) * cell_size, 0)
        end_point = ((i + 1) * cell_size, frame_height)
        color = (255, 255, 255)
        thickness = 1
        image = cv.line(image, start_point, end_point, color, thickness)

    return image


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
        image = cv.resize(image,(frame_width,frame_height))


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
            r, c = get_row_col((x + w) // 2, (y + h) // 2)
            if id in trajectory:
                trajectory[id].append(((x + w) // 2, (y + h) // 2))
            else:
                trajectory[id] = []
                trajectory[id].append(((x + w) // 2, (y + h) // 2))
            cv.rectangle(image, (x, y), (w, h), (255, 0, 255), 2)
            cv.putText(image, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
            heat_matrix[r, c] += 1


        temp_heat_matrix = heat_matrix.copy()
        temp_heat_matrix = resize(temp_heat_matrix, (frame_height, frame_width))
        temp_heat_matrix = temp_heat_matrix / np.max(temp_heat_matrix)
        temp_heat_matrix = np.uint8(temp_heat_matrix * 255)
        

        image_heat = cv.applyColorMap(temp_heat_matrix, cv.COLORMAP_JET)
        image = draw_grid(image)
        cv.addWeighted(image_heat, alpha, image, 1-alpha, 0, image)

        cv.imshow('Image',image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    main(video_path)
