from vidgear.gears import NetGear
import cv2
from matplotlib import pyplot as plt
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor
import numpy as np
import pantilthat as pth
import time
from multiprocessing import Manager
from multiprocessing import Process
import signal
import sys

SERVO_RANGE = (-90, 90)


class PID:
    def __init__(self, kP=1, kI=0, kD=0):
        self.kP = kP
        self.kI = kI
        self.kD = kD
    
    def initialize(self):
        self.curr_ts = time.time()
        self.prev_ts = self.curr_ts
        self.prev_error = 0
        self.cP = 0
        self.cI = 0
        self.cD = 0
        
    def update(self, error, sleep=0.2):
        time.sleep(sleep)
        self.curr_ts = time.time()
        dt = self.curr_ts - self.prev_ts
        de = error - self.prev_error
        self.cP = error
        self.cI += error * dt
        self.cD = de / dt if dt > 0 else 0
        self.prev_ts = self.curr_ts
        self.prev_error = error
        # print(self.kP, self.cP, self.kI, self.cI, self.kD, self.cD)
        return sum([self.kP*self.cP, self.kI*self.cI, self.kD*self.cD])

class Detector:
    def __init__(self, 
                 model='models/lite-model_efficientdet_lite0_detection_metadata_1.tflite', 
                 backSub = cv2.createBackgroundSubtractorMOG2(),
                 area_threshold=50):
        base_options = core.BaseOptions(file_name=model, use_coral=True)
        detection_options = processor.DetectionOptions(
            max_results=3, 
            score_threshold=0.5)
        options = vision.ObjectDetectorOptions(
            base_options=base_options, 
            detection_options=detection_options)
        detector = vision.ObjectDetector.create_from_options(options)
        self.detector = detector
        self.backSub = backSub
        self.background = None
        self.prev_frames = []
        self.area_threshold = area_threshold
        
    def update_background(self, frame):
        if self.background is None:
            self.background = frame
            return
        
        self.background = cv2.accumulateWeighted(frame, self.background.astype(float), 0.5).astype(np.uint8)
    
    def detect_obj(self, frame):
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
        input_tensor = vision.TensorImage.create_from_array(im)
        results = self.detector.detect(input_tensor)
        
        max_area = 0
        center = frame.shape[1]//2, frame.shape[0]//2
        w, h = None, None
        for res in results.detections:
            if res.categories[0].category_name == 'person':
                bb = res.bounding_box
                x = bb.origin_x
                y = bb.origin_y
                w = bb.width
                h = bb.height
                x2 = bb.origin_x + w
                y2 = bb.origin_y + h
                label = f"{res.categories[0].category_name}: {res.categories[0].score}"
                cv2.rectangle(im, (x, y), (x2, y2), color=(0, 255, 0), thickness=3)
                cv2.putText(im, label, (x + 12, y + 12), 0, 1e-3 * im.shape[0], (0, 255, 0))
                
                if w * h > max_area:
                    max_area = w * h
                    center = x+w//2, y+h//2
        return im, (center, w, h)
    
    def get_foreground_mask(self, frame, disp=False):
        mask = self.backSub.apply(frame)
        if disp:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return mask
    
    @staticmethod
    def get_gray_blur(frame):
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        return blurred.astype(np.uint8)
    
    def get_smoothed_frame(self, frame):
        blurred = self.get_gray_blur(frame)
        background = self.get_gray_blur(self.background)
        
        diff = cv2.absdiff(background, blurred)
        threshed = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        dilated = cv2.dilate(threshed, None, iterations=2)
        eroded = cv2.erode(dilated, None, iterations=2)
        return eroded
    
    def detect_mvmt(self, frame, method='backsub'):
        if method == 'backsub':
            processed = self.get_foreground_mask(frame)
        else:
            processed = self.get_smoothed_frame(frame)
        contours = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        max_area = 0
        center = frame.shape[1]//2, frame.shape[0]//2
        w, h = None, None
        largest = None
        for contour in contours:
            # if cv2.contourArea(contour) > self.area_threshold:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w * h > self.area_threshold:
                largest = x, y, w, h
                if w * h > max_area:
                    max_area = w * h
                    center = x+w//2, y+h//2
        if largest:
            x, y, w, h = largest
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (center, w, h)
       
def signal_handler(sig, frame):
	print("[INFO] You pressed `ctrl + c`! Exiting...")
	pth.servo_enable(1, False)
	pth.servo_enable(2, False)
	sys.exit()

def detect_process(objX, objY, centerX, centerY):
    stream = cv2.VideoCapture(0)
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    options = {"flag": 0, "copy": False, "track": False}

    server = NetGear(
        address="192.168.68.122",
        port="5454",
        protocol="tcp",
        pattern=0,
        logging=True,
        **options
    )
    
    def signal_handler(sig, frame):
        print("[INFO] You pressed `ctrl + c`! Exiting...")
        pth.servo_enable(1, False)
        pth.servo_enable(2, False)
        stream.release()
        server.close()
        sys.exit()
    signal.signal(signal.SIGINT, signal_handler)
    
    # model = 'models/lite-model_object_detection_mobile_object_localizer_v1_1_metadata_2.tflite' 
    # model = 'models/squirrel.tflite'
    # model = 'models/mobilenet_v2.tflite'
    model = 'models/lite-model_efficientdet_lite0_detection_metadata_1.tflite'
    backSub = cv2.createBackgroundSubtractorMOG2(history=50, 
                                                 detectShadows=False)
    detector = Detector(model, backSub, 200)
    while True:
        try:
            # read frames from stream
            (grabbed, frame) = stream.read()
            # check for frame if not grabbed
            if not grabbed:
                break
            frame = cv2.flip(frame, -1)
            
            # H, W = frame.shape[:2]
            # centerX.value = W // 2
            # centerY.value = H // 2
            
            detector.update_background(frame)
            # im, pos = detector.detect_mvmt(frame)
            im, pos = detector.detect_obj(frame)
            center, w, h = pos
            print(center)
            objX.value, objY.value = center
            # im = detector.detect_mvmt(frame, method='naive')
            
            server.send(im)
        except KeyboardInterrupt:
            break

def pid_process(output, p, i, d, coord, center, action):
    signal.signal(signal.SIGINT, signal_handler)
    time.sleep(2)
    pid = PID(p.value, i.value, d.value)
    pid.initialize()
    while True:
        error = center.value - coord.value
        output.value = pid.update(error)
        if action == 'pan':
            print(f'{action} error {error} angle: {output.value} coords: {center.value, coord.value} vals: {pid.cP, pid.kP, pid.cI, pid.kI, pid.cD, pid.kD}')
    
def in_range(val, start, end):
    # determine the input value is in the supplied range
    return (val >= start and val <= end)

def set_servos(pan, tlt):
    # signal trap to handle keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)
    # loop indefinitely
    while True:
        # the pan and tilt angles are reversed
        panAngle = 1 * pan.value
        tiltAngle = -1 * tlt.value
        # if the pan angle is within the range, pan
        if in_range(panAngle, SERVO_RANGE[0], SERVO_RANGE[1]):
            pth.pan(panAngle)
        # if the tilt angle is within the range, tilt
        if in_range(tiltAngle, SERVO_RANGE[0], SERVO_RANGE[1]):
            pth.tilt(tiltAngle)
            
with Manager() as manager:
    pth.servo_enable(1, True)
    pth.servo_enable(2, True)
    # set integer values for the object center (x, y)-coordinates
    centerX = manager.Value("i", 160)
    centerY = manager.Value("i", 160)
    # set integer values for the object's (x, y)-coordinates
    objX = manager.Value("i", 0)
    objY = manager.Value("i", 0)
    # pan and tilt values will be managed by independed PIDs
    pan = manager.Value("i", 0)
    tlt = manager.Value("i", 0)
    # set PID values for panning
    panP = manager.Value("f", 0.08)
    panI = manager.Value("f", 0.1)
    panD = manager.Value("f", 0.005)
    # set PID values for tilting
    tiltP = manager.Value("f", 0.15)
    tiltI = manager.Value("f", 0.15)
    tiltD = manager.Value("f", 0.005)
    
    processObjectCenter = Process(target=detect_process, 
                                  args=(objX, objY, centerX, centerY))
    processPanning = Process(target=pid_process, 
                             args=(pan, panP, panI, panD, objX, centerX, 'pan'))
    processTilting = Process(target=pid_process, 
                             args=(tlt, tiltP, tiltI, tiltD, objY, centerY, 'tilt'))
    processSetServos = Process(target=set_servos, args=(pan, tlt))
    # start all 4 processes
    processObjectCenter.start()
    processPanning.start()
    processTilting.start()
    processSetServos.start()
    # join all 4 processes
    processObjectCenter.join()
    processPanning.join()
    processTilting.join()
    processSetServos.join()
    # disable the servos
    pth.servo_enable(1, False)
    pth.servo_enable(2, False)
