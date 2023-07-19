from mediapipe.tasks import python
import mediapipe as mp
import math

"""
def draw_result(img,result):
    for face in result:
        img = cv2.rectangle(img,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,0,255),2)
        for keypoint in list(map(list,np.array_split(face[4:][:-1], 6))):
            img = cv2.circle(img,tuple(keypoint),2,(255,0,255),-1)
    return img
"""

class FaceDetection(object):

    def __init__(self,model_path='face.tflite'):
        
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        # Create a face detector instance with the image mode:
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE)
        
        self.detector = FaceDetector.create_from_options(options)
    
    
    def _result2output(self,detection_result,img_w,img_h):
        """ 
        
        return list(=faces) of list. ex = [[
                x,y,w,h,
                left_eye_x,left_eye_y,
                right_eye_x,right_eye_y,
                nose_x,nose_y,
                lip_x,lip_y,
                left_ear_x,left_ear_y,
                right_ear_x,right_ear_y,
                confidence
            ]]
        """
        output = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            keypoints=[]
            for keypoint in detection.keypoints:
                keypoints.append([
                    min(math.floor(keypoint.x * img_w), img_w - 1),
                    min(math.floor(keypoint.y * img_h), img_h - 1)
                ])
            output.append([
                bbox.origin_x, bbox.origin_y, bbox.width, bbox.height,
                *sum(keypoints,[]),
                detection.categories[0].score
            ])
        return output
        
    def detect(self,img):
        """img:rgb numpy array"""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    
        result = self.detector.detect(mp_image)
    
        return self._result2output(result,mp_image.width,mp_image.height)
