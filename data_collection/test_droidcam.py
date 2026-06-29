import cv2

cap = cv2.VideoCapture(2)  # DroidCam virtual camera at /dev/video2


if not cap.isOpened():
    print("카메라 열기 실패 (index 99)")
    # 다른 인덱스도 시도
    for i in range(10):
        c = cv2.VideoCapture(i)
        if c.isOpened():
            print(f"  -> index {i} 에서 카메라 발견")
            c.release()
else:
    print("DroidCam 연결 성공! 'q' 누르면 종료")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break
        cv2.imshow("DroidCam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
