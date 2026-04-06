import cv2
import numpy as np
import matplotlib.pyplot as plt


#解析の初期設定
deterioration_factor=1.0
threshold=5.0
limit_year = (threshold/deterioration_factor)**2

#カメラの起動
cap=cv2.VideoCapture(0)

print(f'【システム起動】寿命しきい値:{limit_year}年で設定')

while True:
    ret,frame=cap.read()
    if not ret:break

    #リアルタイム解析(錆と苔の抽出)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    frame=cv2.GaussianBlur(frame,(15,15),0)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #茶色の範囲設定
    lower_brown=np.array([5,40,40])
    upper_brown=np.array([25,255,255])
    mask_brown=cv2.inRange(hsv,lower_brown,upper_brown)

    #緑色の範囲設定
    lower_green=np.array([25,50,50])
    upper_green=np.array([85,255,255])
    mask_green=cv2.inRange(hsv,lower_green,upper_green)

    #maskを合体！
    mask_combined=cv2.bitwise_or(mask_brown,mask_green)
    kernel=np.ones((3,3),np.uint8)
    mask=cv2.morphologyEx(mask_combined,cv2.MORPH_OPEN,kernel)
    #合体マスクで元の画像から色を抜き出す
    res=cv2.bitwise_and(frame,frame,mask=mask_combined)

    #占有率の計算
    white_pixels=cv2.countNonZero(mask_combined)
    total_pixels=mask_combined.size
    ratio=(white_pixels/total_pixels)*100
    #映像の上に文字を書き込む（リアルタイム診断）
    text=f"Rust Ratio:{ratio:.2f}%"
    cv2.putText(frame,text,(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("Real-time Diagnosis",frame)

    cv2.imshow("Original",frame)
    cv2.imshow("Combined Mask",mask_combined)
    cv2.imshow("Result(Brown and Green)",res)

    cv2.putText(frame,f"Life:{limit_year}yrs",(20,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),2)
    
    
    key = cv2.waitKey(1) & 0xFF
    if key==ord('s'):
            filename=f"leak_check_{ratio:.1f}percent.jpg"
            cv2.imwrite(filename,frame)
            print('【保存完了】診断結果{filename}を保存しました')
    if key==ord('t'):
            print('【通信命令】ドローンに『Takeoff（離陸）』の信号を送ります...')
            #ここにtello.takeoff()と書くと実機が浮き上がる
    if key==ord('l'):
           print('【通信命令】ドローンに『Landing（着陸）』の信号を送ります...')
           #tello.landing()で着陸する？ 
    if key==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()