import cv2
FileName = './images/seal-hand/seal-hand-0.jpg'
bgrImg1 = cv2.imread(FileName)
FileName = './images/seal-hand/seal-hand-12.jpg'
bgrImg2 = cv2.imread(FileName)

if bgrImg1 is not None:
    grayImg1 = cv2.cvtColor(bgrImg1, cv2.COLOR_BGR2GRAY)
    grayImg2 = cv2.cvtColor(bgrImg2, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray1', grayImg1)
    cv2.imshow('gray2', grayImg2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
