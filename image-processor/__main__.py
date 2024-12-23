import numpy as np
import cv2
import json
from riskDetection import getRiskMatrix, create_display_matrix


def getCalibrationInformation():
    with open("./calibration.json", "r") as file:
        return json.load(file)


def getFramePair():
    foundImageNumber = 0
    framePair = []
    while foundImageNumber < 2:
        foundImageNumber = 0
        framePair = []

        left = cv2.imread("./shared-memory/l-frame.png", cv2.IMREAD_GRAYSCALE)
        if left is not None:
            framePair.append(left)
            foundImageNumber += 1

        right = cv2.imread("./shared-memory/r-frame.png", cv2.IMREAD_GRAYSCALE)
        if right is not None:
            framePair.append(right)
            foundImageNumber += 1

        if foundImageNumber < 2:
            print("Could not find all images. Waiting")

    return framePair


def applyCameraMatrix(framePair, calibration):
    matrixLeft = calibration["cameraMatrixLeft"]
    matrixRight = calibration["cameraMatrixRight"]

    framePair[0] = np.asarray(
        cv2.undistort(framePair[0], np.matrix(matrixLeft), 0),
        np.uint8,
    )
    framePair[1] = np.asarray(
        cv2.undistort(framePair[1], np.matrix(matrixRight), 0),
        np.uint8,
    )

    return framePair


def calculateDisparity(framePair, calibration):
    lImg = framePair[0]
    rImg = framePair[1]

    numDisparities = calibration["numDisparities"] * 16
    numDisparities = 16 if numDisparities <= 0 else numDisparities
    minDisparity = calibration["minDisparity"]
    blockSize = calibration["blockSize"] + (0 if calibration["blockSize"] % 2 else 1)
    P1 = 8 * blockSize ** calibration["P1"]
    P2 = 32 * blockSize ** calibration["P2"]
    disp12_max_diff = calibration["maxDisparities"]
    uniqueness_ratio = calibration["uniqueness"]
    speckle_window_size = calibration["speckleWindowSize"]
    speckle_range = calibration["speckleRange"]
    pre_filter_cap = calibration["preFilterCap"]
    mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

    stereo = cv2.StereoSGBM_create(
        minDisparity=minDisparity,
        numDisparities=numDisparities,
        blockSize=blockSize,
        P1=P1,
        P2=P2,
        disp12MaxDiff=disp12_max_diff,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        preFilterCap=pre_filter_cap,
        mode=mode,
    )

    # stereo = cv2.StereoBM_create(
    #     numDisparities=numDisparities,
    #     blockSize=blockSize
    # )

    disparity = stereo.compute(lImg, rImg)

    min = disparity.min()
    max = disparity.max()
    map = np.uint8(255 * (disparity - min) / (max - min))

    return {"map": map, "min": min, "max": max}


def calculateDepthMap(disparityMap):
    return cv2.applyColorMap(disparityMap, 11)


def resize(img):
    return cv2.resize(img, (640, 480))
    # return img


def displayImage(image, name="Frame"):
    cv2.imshow(name, resize(image))


def displayFramePair(framePair, name="Frame"):
    resized = [resize(framePair[0]), resize(framePair[1])]
    cv2.imshow(name, np.hstack(resized))


def displayImageGrid(imageMatrix, name="Frame"):
    rows = []
    for y in range(len(imageMatrix)):
        row = imageMatrix[y]
        resized = []
        for x in range(len(row)):
            resized.append(resize(row[x]))
        rowImage = np.hstack(resized)
        rows.append(rowImage)
    grid = np.vstack(rows)
    cv2.imshow(name, grid)


calibration = getCalibrationInformation()
while True:
    calibration = getCalibrationInformation()

    framePair = getFramePair()
    displayFramePair(framePair, "Raw")

    # framePair = applyCameraMatrix(framePair, calibration)
    # displayFramePair(framePair, "Corrected")

    disparity = calculateDisparity(framePair, calibration)
    displayImage(cv2.applyColorMap(disparity["map"], cv2.COLORMAP_JET), "Disparity")
    # displayImage(disparity["map"], "Disparity")
    print(disparity["min"], disparity["max"])

    # depthMap = calculateDepthMap(disparity["map"])
    # displayImage(depthMap, "Depth")

    riskMatrix = getRiskMatrix(disparity, calibration)
    print(riskMatrix)

    cv2.imshow(create_display_matrix(np.array(riskMatrix).reshape(3, 3), (300, 300)))

    cv2.waitKey()
    cv2.destroyAllWindows()
