from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import numpy as np
import cv2
import math

# scripts for crop images
def crop_image(img, position):
    def distance(x1,y1,x2,y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))    
    position = position.tolist()
    for i in range(4):
        for j in range(i+1, 4):
            if(position[i][0] > position[j][0]):
                tmp = position[j]
                position[j] = position[i]
                position[i] = tmp
    if position[0][1] > position[1][1]:
        tmp = position[0]
        position[0] = position[1]
        position[1] = tmp

    if position[2][1] > position[3][1]:
        tmp = position[2]
        position[2] = position[3]
        position[3] = tmp

    x1, y1 = position[0][0], position[0][1]
    x2, y2 = position[2][0], position[2][1]
    x3, y3 = position[3][0], position[3][1]
    x4, y4 = position[1][0], position[1][1]

    corners = np.zeros((4,2), np.float32)
    corners[0] = [x1, y1]
    corners[1] = [x2, y2]
    corners[2] = [x4, y4]
    corners[3] = [x3, y3]

    img_width = distance((x1+x4)/2, (y1+y4)/2, (x2+x3)/2, (y2+y3)/2)
    img_height = distance((x1+x2)/2, (y1+y2)/2, (x4+x3)/2, (y4+y3)/2)

    corners_trans = np.zeros((4,2), np.float32)
    corners_trans[0] = [0, 0]
    corners_trans[1] = [img_width - 1, 0]
    corners_trans[2] = [0, img_height - 1]
    corners_trans[3] = [img_width - 1, img_height - 1]

    transform = cv2.getPerspectiveTransform(corners, corners_trans)
    dst = cv2.warpPerspective(img, transform, (int(img_width), int(img_height)))
    return dst

def order_point(coor):
    arr = np.array(coor).reshape([4, 2])
    sum_ = np.sum(arr, 0)
    centroid = sum_ / arr.shape[0]
    theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
    sort_points = arr[np.argsort(theta)]
    sort_points = sort_points.reshape([4, -1])
    if sort_points[0][0] > centroid[0]:
        sort_points = np.concatenate([sort_points[3:], sort_points[:3]])
    sort_points = sort_points.reshape([4, 2]).astype('float32')
    return sort_points

def process_image_to_text(img_path, output_txt='output.txt', ocr_detection=None, ocr_recognition=None):
    try:
        # Read the image from the path
        image_full = cv2.imread(img_path)

        # Check if image is loaded properly
        if image_full is None:
            raise ValueError("Image could not be read.")

        # Perform OCR detection
        det_result = ocr_detection(image_full)
        det_result = det_result['polygons']

        # Open a file to write the OCR results
        with open(output_txt, 'w', encoding='utf-8') as file:
            for i in range(det_result.shape[0]):
                # Order points for cropping
                pts = order_point(det_result[i])
                
                # Crop the image based on detected points
                image_crop = crop_image(image_full, pts)
                
                # Perform OCR recognition on the cropped image
                result = ocr_recognition(image_crop)
                text = result['text']
                # Write the detected text to the file
                if isinstance(text, list):  # Check if the text is a list
                    text = ''.join(text)   # Join all items in the list into a single string

                # Write the processed text to the file
                file.write(text + ' ')

        print(f"OCR process completed and results are saved to '{output_txt}'.")
    except Exception as e:
        # In case of any error during the process, write an empty file
        print(f"An error occurred: {str(e)}")
        with open(output_txt, 'w') as file:
            pass  # Creating an empty file

def save_first_frame(video_path, output_image_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # 读取第一帧
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(output_image_path, frame)
        print(f"First frame saved to {output_image_path}")
        return output_image_path
    else:
        print("Error: Could not read the first frame.")
        return None

def process_video_directory(video_dir, OCR_output_directory, keyframe_output_directory):

    # Initialize the OCR models
    ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
    ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')

    # 有两层文件夹，第一层文件夹为视频文件夹，第二层文件夹为视频文件，使用两个循环分别处理
    for video_folder in os.listdir(video_dir):
        video_folder_path = os.path.join(video_dir, video_folder)
        if os.path.isdir(video_folder_path):
            for video_file in os.listdir(video_folder_path):
                video_file_path = os.path.join(video_folder_path, video_file)
                video_name = os.path.splitext(os.path.basename(video_file_path))[0]
                output_image_path = os.path.join(keyframe_output_directory,video_folder, f"{video_name}_keyframe.jpg")
                output_txt_path = os.path.join(OCR_output_directory, video_folder, f"{video_name}.txt")
                if not os.path.exists(output_image_path):
                    # 新建文件夹
                    if not os.path.exists(os.path.join(keyframe_output_directory,video_folder)):
                        os.makedirs(os.path.join(keyframe_output_directory,video_folder))
                    if not os.path.exists(os.path.join(OCR_output_directory,video_folder)):
                        os.makedirs(os.path.join(OCR_output_directory,video_folder))
                    img_path, output_txt_path = process_single_video(video_file_path, output_image_path=output_image_path, output_txt_path = output_txt_path, ocr_detection=ocr_detection, ocr_recognition=ocr_recognition)
                    print(f"Processed video {video_name}.") 
                else:
                    print(f"Key frame {output_image_path} already exists, skipping.")

def process_single_video(video_path, output_image_path, output_txt_path, ocr_detection, ocr_recognition):

    # Save the first frame of the video
    img_path = save_first_frame(video_path, output_image_path)
    if img_path is None:
        raise Exception("Failed to extract key frame.")

    # Process the image to extract text
    process_image_to_text(img_path, output_txt_path, ocr_detection=ocr_detection, ocr_recognition=ocr_recognition)
    return img_path, output_txt_path

if __name__ == '__main__':
    video_directory = '/root/bishe/end_output/虚假宣传/segment_video_0.8'  # Specify your video directory
    OCR_output_directory = '/root/bishe/end_output/虚假宣传/OCR'
    keyframe_output_directory = '/root/bishe/end_output/虚假宣传/keyframe'  # Specify your output directory
    process_video_directory(video_directory, OCR_output_directory, keyframe_output_directory)
