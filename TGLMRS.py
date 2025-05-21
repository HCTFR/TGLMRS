# 环境依赖：Python 3.8+, OpenCV 4.5+, MediaPipe 0.8+, smtplib，-*- codeing : utf-8 -*-
import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import smtplib
import os
import ssl
from threading import Thread, Lock
from queue import Queue
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.header import Header
from email.utils import formataddr
from ultralytics import YOLO
#pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple
print("""
 **********         ********        **             ****     ****       *******          ********
/////**///         **//////**      /**            /**/**   **/**      /**////**        **////// 
    /**           **      //       /**            /**//** ** /**      /**   /**       /**       
    /**          /**               /**            /** //***  /**      /*******        /*********
    /**          /**    *****      /**            /**  //*   /**      /**///**        ////////**
    /**          //**  ////**      /**            /**   /    /**      /**  //**              /**
    /**           //********       /********      /**        /**      /**   //**       ******** 
    //             ////////        ////////       //         //       //     //       ////////  

                                                                                            @HTR
   """)


#邮件交互设置
recipients=input('请输入接收端的邮箱地址：')

# 邮件报警配置
EMAIL_HOST='smtp.qq.com'
EMAIL_PORT=587
EMAIL_USER=''
EMAIL_PASS=''
if recipients=='1949':
    print('--+--伟大领袖毛主席万岁！全体中国人民万岁！全世界无产者联合起来！--+--')
    print('---这是一个彩蛋，请重新运行程序并输入正确的邮箱地址！----')
else:
    RECIPIENTS=[str(recipients)]
# 初始化MediaPipe姿势识别
mp_pose=mp.solutions.pose
pose=mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 异常动作检测参数
VIOLENT_THRESHOLD=0.8  # 暴力动作阈值
DEPRESSED_THRESHOLD=0.8  # 抑郁状态阈值


class VideoStream:
    """多线程摄像头读取类"""
    def __init__(self, src=0, width=400, height=400):
        self.cap=cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.queue=Queue(maxsize=1)  # 缓冲1帧，避免堆积
        self.lock=Lock()
        self.stopped=False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame=self.cap.read()
            if not ret:
                break
            if self.queue.full():
                self.queue.get()  # 丢弃旧帧
            self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped=True
        self.cap.release()

class BehaviorAnalyzer:
    def __init__(self):
        self.violent_counter=0
        self.depressed_counter=0
        self.leg_counter=0
        self.last_alert_time=0

    def analyze_pose(self, landmarks):
        """分析姿势关键点数据"""
        # 计算身体各部位相对位置
        left_arm_angle=self._calculate_arm_angle(landmarks, 'left')
        right_arm_angle=self._calculate_arm_angle(landmarks, 'right')
        left_leg_angle=self._calculate_leg_angle(landmarks, 'left')
        right_leg_angle=self._calculate_leg_angle(landmarks, 'right')

        head_angle=self._calculate_head_angle(landmarks)

        # 暴力动作检测（快速挥臂）
        if left_arm_angle>120 or right_arm_angle>120:
            self.violent_counter+=1
        else:
            self.violent_counter=max(0, self.violent_counter - 1)

        # 抑郁状态检测（低头静止）
        if head_angle<40 and np.mean([left_arm_angle, right_arm_angle])<60:
            self.depressed_counter+=1
        else:
            self.depressed_counter=max(0, self.depressed_counter - 1)

        # 踢踹动作检测
        if left_leg_angle>80 or right_leg_angle>80:
            self.leg_counter+=1
        else:
            self.leg_counter=max(0, self.violent_counter - 1)

        is_alert = False
        # 触发报警逻辑
        current_time=time.time()
        if current_time - self.last_alert_time>30:  # 30s冷却
            if self.violent_counter>10:
                self.last_alert_time=current_time
                is_alert=True
            elif self.depressed_counter>30:
                self.last_alert_time=current_time
                is_alert=True
            elif self.leg_counter>2:
                self.last_alert_time=current_time
                is_alert=True
        return is_alert

    def _calculate_arm_angle(self, landmarks, side):
        """计算手臂角度"""
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER if side=='left'
        else mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow=landmarks[mp_pose.PoseLandmark.LEFT_ELBOW if side=='left'
        else mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist=landmarks[mp_pose.PoseLandmark.LEFT_WRIST if side=='left'
        else mp_pose.PoseLandmark.RIGHT_WRIST]

        # 计算向量角度
        vec1=np.array([elbow.x-shoulder.x,elbow.y-shoulder.y])
        vec2=np.array([wrist.x-elbow.x, wrist.y-elbow.y])
        return np.degrees(np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))))

    def _calculate_leg_angle(self, landmarks, side):
        hip=landmarks[mp_pose.PoseLandmark.LEFT_HIP if side=='left'
        else mp_pose.PoseLandmark.RIGHT_HIP]
        knee=landmarks[mp_pose.PoseLandmark.LEFT_KNEE if side=='left'
        else mp_pose.PoseLandmark.RIGHT_KNEE]
        ankle=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE if side=='left'
        else mp_pose.PoseLandmark.RIGHT_ANKLE]
        # 计算向量角度
        vec3=np.array([hip.x-knee.x,hip.y-knee.y])
        vec4=np.array([ankle.x-knee.x,ankle.y-knee.y])
        return np.degrees(np.arccos(np.dot(vec3, vec4)/(np.linalg.norm(vec3) * np.linalg.norm(vec4))))


    def _calculate_head_angle(self, landmarks):
        """计算头部倾斜角度"""
        # 获取关键点
        nose=landmarks[mp_pose.PoseLandmark.NOSE]
        left_ear=landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear=landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

        # 计算耳朵连线与水平线的夹角
        ear_vec=np.array([right_ear.x - left_ear.x, right_ear.y - left_ear.y])
        horizontal_vec=np.array([1, 0])  # 水平基准向量

        # 计算夹角（弧度）
        cosine_angle=np.dot(ear_vec, horizontal_vec)/(
                np.linalg.norm(ear_vec)*np.linalg.norm(horizontal_vec)
        )
        angle=np.degrees(np.arccos(cosine_angle))

        # 计算鼻子与肩膀的相对位置（检测低头）
        left_shoulder=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder=landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_avg_y=(left_shoulder.y + right_shoulder.y)/2
        head_drop=nose.y-shoulder_avg_y  # 值越大表示低头越明显

        # 综合判断（示例逻辑）
        return max(angle, head_drop * 100)  # 将头部位移转换为角度近似值

    def model_alert(self,res):
        """使用自训练模型进行行为检测"""
        if res[0].boxes.cls.tolist():
            is_alert=True
        return is_alert

    def send_alert(self, message, image_path=None):
        """发送HTML格式邮件并内嵌图片"""
        try:
            # 创建多部分邮件（使用related类型支持内嵌资源）
            msg=MIMEMultipart('related')

            # 编码邮件头（支持中文）
            msg['Subject'] = Header('行为预警通知', 'utf-8').encode()
            msg['From'] = formataddr((
                Header('TG肢体监测及响应系统', 'utf-8').encode(),
                EMAIL_USER
            ))
            msg['To']=', '.join(RECIPIENTS)

            # 创建alternative容器用于文本/HTML内容
            msg_alternative=MIMEMultipart('alternative')
            msg.attach(msg_alternative)

            # HTML版本（带内嵌图片）
            html_content=f"""
            <html>
                <body style="font-family: 'Microsoft YaHei', sans-serif; color: #333;">
                    <h2 style="color: #dc3545;">⚠️ 行为预警通知</h2>
                    <p>检测到以下异常行为：</p>
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 5px;">
                        <p>{message}</p>
                        <p style="font-size: 0.9em; color: #666;">
                            检测时间：{time.strftime("%Y-%m-%d %H:%M:%S")}
                        </p>
                    </div>
                    {f'<img src="cid:alert_image" style="max-width: 600px; margin-top: 20px; border: 1px solid #ddd; border-radius: 4px;">' if image_path else ''}
                    <div style="margin-top: 20px; color: #6c757d; font-size: 0.8em;">
                        <hr>
                        <p>此邮件为自动发送，请勿直接回复</p>
                        <p>TG肢体监测及响应系统</p>
                    </div>
                </body>
            </html>
            """
            html_part=MIMEText(html_content, 'html', 'utf-8')
            msg_alternative.attach(html_part)

            # 添加内嵌图片
            if image_path:
                with open(image_path, 'rb') as f:
                    img_data=f.read()

                # 创建图片部分并设置Content-ID
                img_part=MIMEImage(img_data, name=os.path.basename(image_path))
                img_part.add_header('Content-ID', '<alert_image>')
                img_part.add_header('Content-Disposition', 'inline', filename=os.path.basename(image_path))
                msg.attach(img_part)

            # SMTP连接配置
            context=ssl.create_default_context()
            if EMAIL_PORT==465:
                with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT, context=context, timeout=20) as server:
                    server.login(EMAIL_USER, EMAIL_PASS)
                    server.send_message(msg)
            elif EMAIL_PORT==587:
                with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT, timeout=20) as server:
                    server.starttls(context=context)
                    server.login(EMAIL_USER, EMAIL_PASS)
                    server.send_message(msg)

            print("HTML预警邮件发送成功")
        except smtplib.SMTPResponseException as e:

            if e.smtp_code==-1 and e.smtp_error == b'\x00\x00\x00':

                print("邮件已成功发送。")

            else:

                print(f"SMTP响应异常: {e}")

        except Exception as e:

            print(f"发送邮件时发生错误: {e}")


def main():
    pTime=0
    analyzer=BehaviorAnalyzer()
    cv2.namedWindow('TG Limb Monitoring and Response System(TGLMRS)', cv2.WINDOW_NORMAL)
    cap=cv2.VideoCapture(0)  # 使用摄像头
    # 导入自训练模型
    model=YOLO(r'D:\python\pythonProject1\project\AI\study\YOLO\runs\detect\train8\weights\best.pt')
    # 设置视频帧大小
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

    # 初始化视频流
    stream=VideoStream(src=0, width=400, height=400).start()
    time.sleep(1.0)

    # 初始化MediaPipe姿势识别
    mpPose=mp.solutions.pose
    pose=mpPose.Pose()
    #初始化MediaPipe绘图工具
    mpDraw=mp.solutions.drawing_utils

    while cap.isOpened():
        frame = stream.read()
        imgRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)    #将图片转化成RGB格式
        results=pose.process(imgRGB)    #处理RGB格式的图片
        # 在框架上运行 YOLOv8 推理
        res=model(frame, conf=0.55, augment=True)
        # 在框架上可视化结果
        frame=res[0].plot()

        if results.pose_landmarks:      #如果检测到了姿势关键点,就绘制关键点和连接线
            mpDraw.draw_landmarks(frame,results.pose_landmarks,mpPose.POSE_CONNECTIONS)     #绘制关键点和连接线
            for id,lm in enumerate(results.pose_landmarks.landmark):
                h,w,c=frame.shape
                _,cy=int(lm.x * w),int(lm.y * h)
            if analyzer.analyze_pose(results.pose_landmarks.landmark):
                # 保存触发时的画面
                timestamp=time.strftime("%Y%m%d_%H%M%S")
                try:
                    cv2.imwrite(os.path.join(f"alert_{timestamp}.jpg"), frame)
                except Exception as e:
                    print(f"文件保存失败：{str(e)}")
                # 发送带附件的邮件
                analyzer.send_alert("Unusual behavior detected! See the attached screenshot for details",
                                    f"alert_{timestamp}.jpg"
                                    )
        #绘制FPS
        cTime=time.time()
        fps=1/(cTime - pTime)
        pTime=cTime
        cv2.putText(frame, f'FPS:{int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)


        # 显示实时画面
        cv2.imshow('TG Limb Monitoring and Response System(TGLMRS)',frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
