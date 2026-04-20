import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
import tempfile
import datetime
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


#定数・設定
MODEL_PATH='runs/detect/train32/weights/best.pt'
WINDOW_TITLE="コンクリート解析ツールプロトタイプ"
WINDOW_SIZE="1100x720"
DETERIORATION_FACTOR=1.0
THRESHOLD=5.0

#劣化判定の閾値
RUST_WARN=10.0 #%
RUST_DENGER=25.0 #%
CRACK_WARN=1 #個数
CRACK_DENGER=3 #個数

#メインアプリクラス
class ConcreteAnalyzerApp:
    def __init__(self,root):
        self.root=root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg="#1e1e1e")

        #状態変数
        self.model=None
        self.current_image_path=None
        self.cap=None
        self.camera_running=False
        self.camera_thread=None
        self.limit_year=int((THRESHOLD/DETERIORATION_FACTOR)**2)

        #UI構築
        self._build_ui()

        #モデル読み込み(バックグラウンド)
        threading.Thread(target=self._load_model,daemon=True).start()
    
    #UI構築
    def _build_ui(self):
        
        style=ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",background="#2d2d2d",borderwidth=0)
        style.configure("TNotebook.Tab",background="#2d2d2d",foreground="#aaaaaa",
                        padding=[14,6],font=("Helvetica",10))
        style.map("TNotebook.Tab",
                  background=[("selected","#3c3c3c")],
                  foreground=[("selected","#ffffff")])
        style.configure("TFrame",background="#2d2d2d")
        style.configure("TLabel",background="#2d2d2d",foreground="#cccccc",font=("Helevetica",10))
        style.configure('Title.TLabel',background="#2d2d2d",foreground="#ffffff",
                        font=("Helvetica,10"),padding=[10,5],borderwidth=0)
        style.map("Tbutton",
                  background=[("active","#505050")])
        style.configure("Accent.TButton",background="#1a6fbc",foreground="#ffffff",
                        font=("HElvetica",10,"bold"),padding=[10,5],borderwidth=0)
        style.map("Accent.TButton",
                  background=[("active",'#1e85e0')])
        style.configure("Danger.TLabel",background="#2d2d2d",foreground="#e05555",
                        font=("Helvetica",10,"bold"))
        style.configure("Warn.TLabel",background="#2d2d2d",foreground="#e0a020",
                        font=("Helvetica",10,'bold'))
        style.configure("OK.TLabel",background="#2d2d2d",foreground="#50c878",
                        font=("Helvetica",10,"bold"))
        
        #タイトルバー
        title_bar=tk.Frame(self.root,bg="#111111",height=40)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)
        tk.Label(title_bar,text=WINDOW_TITLE,bg="#111111",fg="#cccccc",
                 font=("Helvetica",11,"bold")).pack(side=tk.LEFT,padx=16,pady=8)
        
        #ステータスバー(下部)
        self.status_var=tk.StringVar(value="起動中...モデルを読み込んでいます")
        status_bar=tk.Frame(self.root,bg="#111111",height=26)
        status_bar.pack(side=tk.BOTTOM,fill=tk.X)
        status_bar.pack_propagate(False)
        tk.Label(status_bar,textvariable=self.status_var,bg="#111111",fg="#888888",
                 font=("Helvetica",9)).pack(side=tk.LEFT,padx=12,pady=4)
        self.model_status_label=tk.Label(status_bar,text="・モデル未ロード",
                                          bg="#111111",fg="#e05555",font=("Helvetica,9"))
        self.model_status_label.pack(side=tk.RIGHT,padx=12,pady=4)

        #タブ
        self.notebook=ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH,expand=True,padx=0,pady=0)

        tab1=ttk.Frame(self.notebook)
        tab2=ttk.Frame(self.notebook)
        tab3=ttk.Frame(self.notebook)
        tab4=ttk.Frame(self.notebook)
        self.notebook.add(tab1,text="　画像解析　")
        self.notebook.add(tab2,text="　リアルタイム　")
        self.notebook.add(tab3,text="　劣化予測　")
        self.notebook.add(tab4,text="　使い方　")

        self._build_tab_image(tab1)
        self._build_tab_realtime(tab2)
        self._build_tab_prediction(tab3)
        self._build_tab_help(tab4)


        #タブ１：画像解析
    def _build_tab_image(self,parent):
        #左パネル：入力＋ボタン
        left=tk.Frame(parent,bg="#2d2d2d",width=340)
        left.pack(side=tk.LEFT,fill=tk.Y,padx=0,pady=0)
        left.pack_propagate(False)

        tk.Label(left,text="入力画像",bg="#2d2d2d",fg="#888888",
                 font=("Helvetica",9)).pack(anchor=tk.W,padx=14,pady=(14,4))
        
        #サムネイル
        self.thumb_frame=tk.Label(left,bg="#1a1a1a",height=12,
                                  text="画像を開くボタンで選択\n(.jpg/.png/.webp)",
                                  fg="#555555",font=("Helvetica",10),
                                  relief=tk.FLAT,bd=0)
        self.thumb_frame.pack(fill=tk.X,padx=14,pady=0)

        #ボタン群
        btn_frame=tk.Frame(left,bg="#2d2d2d")
        btn_frame.pack(fill=tk.X,padx=14,pady=10)
        ttk.Button(btn_frame,text="画像を開く",command=self._open_image).pack(fill=tk.X,pady=3)
        ttk.Button(btn_frame,text="解析実行",command=self._run_analysis,
                   style="Accent.TButton").pack(fill=tk.X,pady=3)
        ttk.Button(btn_frame,text="PDFレポート保存",command=self._save_report).pack(fill=tk.X,pady=3)

        #区切り
        tk.Frame(left,bg="#3a3a3a",height=1).pack(fill=tk.X,padx=14,pady=6)

        #解析結果サマリー
        tk.Label(left,text="解析サマリー",bg="#2d2d2d",fg="#888888",
                 font=("Helvetica",9)).pack(anchor=tk.W,padx=14,pady=(4,6))
        
        self.crack_var=tk.StringVar(value="-")
        self.rust_var =tk.StringVar(value="-")
        self.moss_var =tk.StringVar(value="-")
        self.life_var =tk.StringVar(value="-")

        for label_text,var,color_var_name in [
            ("ひび割れ検出数",self.crack_var,"crack_label"),
            ("錆 占有率",self.rust_var,"rust_label"),
            ("苔 占有率",self.moss_var,"moss_label"),
        ]:
            row=tk.Frame(left,bg="#252525")
            row.pack(fill=tk.X,padx=14,pady=1)
            tk.Label(row,text=label_text,bg="#252525",fg="#888888",
                     font=("Helvetica",9),width=14,anchor=tk.W).pack(side=tk.LEFT,padx=8,pady=4)
            lbl=tk.Label(row,textvariable=var,bg="#252525",fg="#cccccc",
                         font=("Helvetica",11,"bold"),anchor=tk.E)
            lbl.pack(side=tk.RIGHT,padx=8)
            setattr(self,color_var_name,lbl)
        
        tk.Label(left,textvariable=self.life_var,bg="#2d2d2d",fg="#50c878",
                 font=("Helvetica",11,"bold")).pack(anchor=tk.W,padx=14,pady=(10,4))
        

        #ログ
        tk.Label(left,text="ログ",bg="#2d2d2d",fg="#888888",
                 font=("Helvetica",9)).pack(anchor=tk.W,padx=14,pady=(6,2))
        self.log_text=tk.Text(left,bg="#111111",fg="#aaaaaa",font=("Courier",9),
                              height=8,state=tk.DISABLED,relief=tk.FLAT,
                              insertbackground="#cccccc")
        self.log_text.pack(fill=tk.BOTH,expand=True,padx=14,pady=(0,14))
        self.log_text.tag_config("ok",foreground="#50c878")
        self.log_text.tag_config("warn", foreground="#e0a020")
        self.log_text.tag_config("err",foreground="#e05555")

        #右パネル：結果画像
        right=tk.Frame(parent,bg="#1a1a1a")
        right.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)
        tk.Label(right,text="AI解析結果",bg="#1a1a1a",fg="#888888",
                 font=("Helvetica",9)).pack(anchor=tk.W,padx=14,pady=(14,4))
        
        self.result_label=tk.Label(right,bg="#111111",
                                   text="解析結果がここに表示されます",
                                   fg="#444444",font=("Helvetica",11))
        self.result_label.pack(fill=tk.BOTH,expand=True,padx=14,pady=(0,14))

        #タブ2：リアルタイム

    def _build_tab_realtime(self,parent):
        left=tk.Frame(parent,bg="#2d2d2d",width=240)
        left.pack(side=tk.LEFT,fill=tk.Y)
        left.pack_propagate(False)

        tk.Label(left,text="リアルタイム解析",bg="#2d2d2d",fg="#888888",
                 font=("Helvetica",9)).pack(anchor=tk.W,padx=14,pady=(14,8))
        
        self.cam_btn=ttk.Button(left,text="カメラ起動",command=self._toggle_camera,
                                style="Accent.TButton")
        self.cam_btn.pack(fill=tk.X,padx=14,pady=4)
        ttk.Button(left,text="スナップショット保存",command=self._save_snapshot).pack(
            fill=tk.X,padx=14,pady=4)
        
        tk.Frame(left,bg="#3a3a3a",height=1).pack(fill=tk.X,padx=14,pady=10)

        tk.Label(left,text="リアルタイム診断",bg="#2d2d2d",fg="#888888",
                 font=("Helvetica",9)).pack(anchor=tk.W,padx=14,pady=(0,6))
        
        self.rt_rust_var=tk.StringVar(value="錆: -")
        self.rt_moss_var=tk.StringVar(value="苔: -")
        tk.Label(left,textvariable=self.rt_rust_var,bg="#2d2d2d",fg="#e0a020",
                 font=("Helvetica",12,"bold")).pack(anchor=tk.W,padx=14,pady=2)
        tk.Label(left,textvariable=self.rt_moss_var,bg="#2d2d2d",fg="#50c878",
                 font=("Helvetica",12,"bold")).pack(anchor=tk.W,padx=14,pady=2)
        
        right=tk.Frame(parent,bg="#111111")
        right.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)
        self.cam_label=tk.Label(right,bg="#111111",
                                text="カメラ起動ボタンを押してください",
                                fg="#444444",font=("Helvetica",11))
        self.cam_label.pack(fill=tk.BOTH,expand=True)
        self.cam_frame=None

        #劣化予測グラフ
    def _build_tab_prediction(self,parent):
        ctrl=tk.Frame(parent,bg="#2d2d2d",width=260)
        ctrl.pack(side=tk.LEFT,fill=tk.Y)
        ctrl.pack_propagate(False)

        tk.Label(ctrl,text="劣化パラメータ",bg="#2d2d2d",fg="#888888",
                 font=("Helvetica",9)).pack(anchor=tk.W,padx=14,pady=(14,6))
        
        for label_text,from_,to_,init_,attr_name in [
            ("劣化係数",0.5,3.0,1.0,"det_factor"),
            ("限界深さ(cm)",1.0,10.0,5.0,"threshold_val"),
        ]:
            tk.Label(ctrl,text=label_text,bg="#2d2d2d",fg="#aaaaaa",
                     font=("Helvetica",9)).pack(anchor=tk.W,padx=14,pady=(8,0))
            var=tk.DoubleVar(value=init_)
            setattr(self,attr_name+"_var",var)
            val_label=tk.Label(ctrl,text=str(init_),bg="#2d2d2d",fg="#ffffff",
                               font=("Helvetica",10,"bold"))
            val_label.pack(anchor=tk.E,padx=14)

            def make_cmd(v=var, vl=val_label, an=attr_name):
                def cmd(val):
                    vl.config(text=f"{float(val):.1f}")
                    self._update_graph()
                return cmd
            
            sl=tk.Scale(ctrl, variable=var,from_=from_,to=to_,resolution=0.1,
                        orient=tk.HORIZONTAL,bg="#2d2d2d",fg="#aaaaaa",
                        troughcolor="#1a1a1a",activebackground="#1a6fbc",
                        highlightthickness=0,bd=0,command=make_cmd())
            sl.pack(fill=tk.X,padx=14,pady=(0,4))

        tk.Frame(ctrl,bg="#3a3a3a",height=1).pack(fill=tk.X,padx=14,pady=10)

        self.pred_life_var=tk.StringVar(value=f"推定寿命:{self.limit_year}年")
        tk.Label(ctrl,textvariable=self.pred_life_var,bg="#2d2d2d",fg="#50c878",
                 font=("Helvetica",13,"bold")).pack(anchor=tk.W,padx=14,pady=4)
        
        ttk.Button(ctrl,text="グラフを更新",command=self._update_graph).pack(
            fill=tk.X,padx=14,pady=10)
        
        #グラフエリア
        graph_frame=tk.Frame(parent,bg="#1a1a1a")
        graph_frame.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)

        fig,ax=plt.subplots(figsize=(7,4.5))
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#111111")
        self.fig=fig
        self.ax=ax

        self.graph_canvas=FigureCanvasTkAgg(fig,master=graph_frame)
        self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True,padx=10,pady=10)
        self._update_graph()
    
    #タブ４：使い方
    def _build_tab_help(self,parent):
        frame=tk.Frame(parent,bg="#2d2d2d")
        frame.pack(fill=tk.BOTH,expand=True,padx=24,pady=20)

        help_text="""【コンクリート解析ツールプロトタイプ　使い方】
■　画像解析タブ
　１．「画像を開く」ボタンでコンクリート画像を選択（.jpg/.png/,webp）
　２．「解析実行」ボタンを押すと YOLOv8 でひび割れ検出 + 錆・苔の色解析が走ります
　３．右側に解析結果画像、左側にサマリー数値が表示されます
　４．「ＰＤＦレポート保存」でＪＰＧとして診断結果を保存できます

■　リアルタイムタブ
　１．「カメラ起動」ボタンでＰＣカメラからリアルタイム映像取得
　２．錆・苔の占有率がリアルタイムで表示されます
　３．「スナップショット保存」で現在フレームを保存

■　劣化予測タブ
　- 劣化係数・限界深さのスライダーで５０年間の劣化グラフが変化します
　- 推定寿命（年数）が自動計算されます

■　モデルについて
　- モデルパス: runs/detect/train30/weichts/best.pt
　- 学習データ: コンクリートひび割れ画像 約18,000枚
　- 精度m mAP50: 0.91

■　注意事項
　- モデルファイル（best.pt）がこのスクリプトと同じフォルダ構造にある必要があります
　- カメラが接続されていない場合はリアルタイム機能は動作しません
"""

        text_widget=tk.Text(frame,bg="#1a1a1a",fg="#aaaaaa",
                            font=("Courier",10),relief=tk.FLAT,wrap=tk.WORD)
        text_widget.insert(tk.END,help_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH,expand=True)

    #ロジック：モデル読み込み
    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model=model=YOLO(MODEL_PATH)
            self.root.after(0,lambda:self._set_status(f"モデル読み込み完了:{MODEL_PATH}",ok=True))
            self.root.after(0,lambda:self._log(f"モデル読み込み完了({MODEL_PATH})","ok"))
        except Exception as e:
            self.root.after(0,lambda:self._set_status(f"モデル読み込み失敗:{e}",ok=False))
            self.root.after(0,lambda:self._log(f"モデル読み込み失敗;{e}","err"))

    def _set_status(self,msg,ok=True):
        self.status_var.set(msg)
        if ok:
            self.model_status_label.config(text=f"●{os.path.basename(MODEL_PATH)}",fg="#50c878")
        else:
            self.model_status_label.config(text="●モデルエラー",fg="#e05555")

    #ロジック：画像を開く
    def _open_image(self):
        path=filedialog.askopenfilename(
            title="解析する画像を選択",
            filetypes=[("画像ファイル","*.jpg*.jpeg*.png*.webp"),("すべて","*.*")]
        )    
        if not path:
            return
        self.current_image_path=path
        self._log(f"画像を読み込みました:{os.path.basename(path)}","ok")

        #サムネイル表示
        img=Image.open(path)
        img.thumbnail((310,195))
        photo=ImageTk.PhotoImage(img)
        self.thumb_frame.config(image=photo,text="")
        self.thumb_frame.image=photo

    #ロジック：解析実行
    def _run_analysis(self):
        if self.current_image_path is None:
            messagebox.showwarning("画像未選択","先に「画像を開く」で画像を選択してください。")
            return
        if self.model is None:
            messagebox.showerror("モデル未ロード","モデルの読み込みが完了していません。少し待ってから再度お試しください。")
            return
        self._log("解析開始．．．","ok")
        threading.Thread(target=self._analyze_thread,daemon=True).start()
    
    def _analyze_thread(self):
        try:
            img_bgr=cv2.imread(self.current_image_path)
            if img_bgr is None:
                self.root.after(0,lambda:self._log("画像の読み込みに失敗しました","err"))
                return
            
            #YOLOv8　ひび割れ検出
            results=self.model(img_bgr)
            annotated=results[0].plot()

            # クラス別カウント
            class_counts={}
            for r in results:
                for box in r.boxes:
                    cls_id=int(box.cls[0])
                    cls_name=self.model.names[cls_id]
                    class_counts[cls_name]=class_counts.get(cls_name,0)+1
            crack_count =class_counts.get("crack",0)
            repair_count=class_counts.get("repair",0)
            seam_count  =class_counts.get("seam",0)
            total_count =sum(class_counts.values())
            #錆・苔 色検出
            blurred=cv2.GaussianBlur(img_bgr,(15,15),0)
            hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
            lower_brown=np.array([5,40,40])
            upper_brown=np.array([25,255,255])
            mask_brown =cv2.inRange(hsv,lower_brown,upper_brown)

            lower_green=np.array([25,40,40])
            upper_green=np.array([85,255,255])
            mask_green =cv2.inRange(hsv,lower_green,upper_green)

            kernel= np.ones((3,3),np.uint8)
            mask_rust=cv2.morphologyEx(mask_brown,cv2.MORPH_OPEN,kernel)
            mask_moss=cv2.morphologyEx(mask_green,cv2.MORPH_OPEN,kernel)

            total_px=img_bgr.shape[0]*img_bgr.shape[1]
            rust_ratio=(cv2.countNonZero(mask_rust)/total_px)*100
            moss_ratio=(cv2.countNonZero(mask_moss)/total_px)*100

            #結果をUIスレッドで反映
            self.root.after(0,lambda:self._update_results(
                annotated,crack_count,repair_count,seam_count,total_count,rust_ratio,moss_ratio))
        
        except Exception as e:
            self.root.after(0,lambda:self._log(f"解析エラー:{e}","err"))
    
    def _update_results(self,annotated_bgr,crack_count,repair_count,seam_count,total_count,rust_ratio,moss_ratio):
        #結果画像
        rgb=cv2.cvtColor(annotated_bgr,cv2.COLOR_BGR2RGB)
        pil_img=Image.fromarray(rgb)
        w =self.result_label.winfo_width()or 640
        h =self.result_label.winfo_height()or 480
        pil_img.thumbnail((max(w - 20,200),max(h - 20,200)))
        photo=ImageTk.PhotoImage(pil_img)
        self.result_label.config(image=photo,text="")
        self.result_label.image=photo
        self._last_annotated=annotated_bgr

        # クラス別カウントを保存（PDF用）
        self._crack_count =crack_count
        self._repair_count=repair_count
        self._seam_count  =seam_count
        self._total_count =total_count

        #サマリー更新
        self.crack_var.set(f"{crack_count}個")
        self.rust_var.set(f"{rust_ratio:.1f}%")
        self.moss_var.set(f"{moss_ratio:.1f}%")

        #色分け
        self.crack_label.config(
            fg="#e05555"if crack_count>=CRACK_DENGER else
            "#e0a020" if crack_count>=CRACK_WARN else "#50c878")
        self.rust_label.config(
            fg="#e05555"if rust_ratio>=RUST_DENGER else
            "#e0a020" if rust_ratio>=RUST_WARN else "#50c878")
        self.moss_label.config(
            fg="#e05555" if moss_ratio > 5.0 else "#50c878")
        
        #ログ
        self._log(f"解析完了 - ひび割れ:{crack_count} / 補修跡:{repair_count} / 目地:{seam_count}","ok")
        self._log(f"錆占有率:{rust_ratio:.1f}%　苔占有率: {moss_ratio:.1f}%",
                  "warn" if rust_ratio>=RUST_WARN else "ok")
        if crack_count>=CRACK_DENGER:
            self._log("要点検: ひび割れが多数検出されました","err")

    #ロジック：リアルタイムカメラ
    def _toggle_camera(self):
        if not self.camera_running:
            self.camera_running=True
            self.cam_btn.config(text="カメラ停止")
            self.cap=cv2.VideoCapture(0)
            self.camera_thread=threading.Thread(target=self._camera_loop,daemon=True)
            self.camera_thread.start()
            self._log("カメラ起動","ok")
        else:
            self.camera_running=False
            self.cam_btn.config(text="カメラ起動")
            if self.cap:
                self.cap.release()
            self._log("カメラ停止","warn")
            self.cam_label.config(image="",text="カメラ起動ボタンを押してください", fg="#444444")
    
    def _camera_loop(self):
        while self.camera_running:
            ret,frame=self.cap.read()
            if not ret:
                break

            #錆・苔リアルタイム
            blurred=cv2.GaussianBlur(frame,(15,15),0)
            hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
            mask_rust=cv2.inRange(hsv,np.array([5,40,40]),np.array([25,255,255]))
            mask_moss=cv2.inRange(hsv,np.array([25,50,50]),np.array([85,255,255]))
            total=frame.shape[0]*frame.shape[1]
            r_ratio=(cv2.countNonZero(mask_rust)/total)*100
            m_ratio=(cv2.countNonZero(mask_moss)/total)*100

            #テキスト描画
            cv2.putText(frame,f"Rust:{r_ratio:.1f}%",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,165,255),2)
            cv2.putText(frame,f"Moss:{m_ratio:.1f}%",(10,65),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,220,80),2)
            
            self.cam_frame=frame.copy()

            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            pil_img=Image.fromarray(rgb)
            pil_img.thumbnail((760,520))
            photo=ImageTk.PhotoImage(pil_img)

            self.root.after(0,lambda p=photo, r=r_ratio, m=m_ratio:self._update_cam(p,r,m))
    
    def _update_cam(self,photo,r_ratio,m_ratio):
        self.cam_label.config(image=photo,text="")
        self.cam_label.image=photo
        self.rt_rust_var.set(f"錆: {r_ratio:.1f}%")
        self.rt_moss_var.set(f"苔:{m_ratio:.1f}%")
    
    def _save_snapshot(self):
        if self.cam_frame is None:
            messagebox.showinfo("情報","カメラを起動してください。")
            return
        path=filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG","*.jpg"),("PNG","*.png")],
            initialfile="snapshot.jpg"
        )
        if path:
            cv2.imwrite(path,self.cam_frame)
            self._log(f"スナップショット保存: {os.path.basename(path)}","ok")
    
    #ロジック：劣化予測グラフ
    def _update_graph(self):
        det_factor=self.det_factor_var.get()
        threshold =self.threshold_val_var.get()
        limit_year=int((threshold/det_factor)**2)
        self.limit_year=limit_year
        self.pred_life_var.set(f"推定寿命: {limit_year}年")
        self.life_var.set(f"推定寿命: {limit_year}年")

        years=np.linspace(0,50,1000)
        depth=det_factor*np.sqrt(years)

        ax=self.ax
        ax.clear()
        ax.set_facecolor("#111111")
        ax.tick_params(color="#888888")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        
        ax.plot(years,depth,color="#e05555",linewidth=2,label="劣化深さ")
        ax.axhline(threshold,color="#888888",linestyle="--",linewidth=1,
                   label=f"限界({threshold:.1f}cm)")
        ax.axvline(limit_year,color="#378ADD",linestyle=":",linewidth=1.5,
                   label=f"寿命{limit_year}年")
        ax.text(limit_year+1,threshold*0.3,f"寿命:{limit_year}年",
                color="#378ADD",fontsize=9)
        
        ax.set_xlabel("年数",color="#888888",fontsize=10)
        ax.set_ylabel("中性化深さ(cm)",color="#888888",fontsize=10)
        ax.set_title("コンクリート劣化予測(50年)",color="#cccccc",fontsize=11)
        ax.set_xticks(np.arange(0,55,5))
        ax.legend(facecolor="#1a1a1a",labelcolor="#aaaaaa",edgecolor="#333333",fontsize=9)
        ax.grid(True,color="#222222",linewidth=0.5)

        self.graph_canvas.draw()

        #ロジック：PDFレポート保存
    def _save_report(self):
        if not hasattr(self,"_last_annotated"):
            messagebox.showinfo("情報","先に解析を実行してください。")
            return

        path=filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF","*.pdf")],
            initialfile=f"concrete_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        if not path:
            return

        try:
            self._generate_pdf(path)
            self._log(f"PDFレポート保存: {os.path.basename(path)}","ok")
            messagebox.showinfo("保存完了",f"PDFレポートを保存しました:\n{path}")
        except Exception as e:
            self._log(f"PDF生成エラー: {e}","err")
            messagebox.showerror("エラー",f"PDF生成に失敗しました:\n{e}")

    def _generate_pdf(self, path):
        # 日本語フォント登録（Windowsの場合）
        font_candidates = [
            "C:/Windows/Fonts/meiryo.ttc",
            "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/YuGothM.ttc",
        ]
        jp_font = "Helvetica"  # フォールバック
        for f in font_candidates:
            if os.path.exists(f):
                try:
                    pdfmetrics.registerFont(TTFont("JPFont", f))
                    jp_font = "JPFont"
                    break
                except:
                    continue

        # ページ設定
        doc = SimpleDocTemplate(
            path,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=20*mm,
        )

        # スタイル
        styles = getSampleStyleSheet()
        style_title = ParagraphStyle(
            "title", fontName=jp_font, fontSize=18, leading=24,
            textColor=colors.HexColor("#1a1a1a"), spaceAfter=4*mm,
        )
        style_subtitle = ParagraphStyle(
            "subtitle", fontName=jp_font, fontSize=10, leading=14,
            textColor=colors.HexColor("#666666"), spaceAfter=6*mm,
        )
        style_section = ParagraphStyle(
            "section", fontName=jp_font, fontSize=12, leading=16,
            textColor=colors.HexColor("#1a6fbc"), spaceBefore=6*mm, spaceAfter=3*mm,
        )
        style_body = ParagraphStyle(
            "body", fontName=jp_font, fontSize=10, leading=15,
            textColor=colors.HexColor("#333333"),
        )
        style_warn = ParagraphStyle(
            "warn", fontName=jp_font, fontSize=11, leading=15,
            textColor=colors.HexColor("#cc3300"), spaceBefore=4*mm,
        )

        # 判定ロジック
        crack_count  = getattr(self, "_crack_count",  0)
        repair_count = getattr(self, "_repair_count", 0)
        seam_count   = getattr(self, "_seam_count",   0)
        total_count  = getattr(self, "_total_count",  0)
        rust_ratio   = float(self.rust_var.get().replace("%","").strip()) if self.rust_var.get() != "-" else 0.0
        moss_ratio   = float(self.moss_var.get().replace("%","").strip()) if self.moss_var.get() != "-" else 0.0
        limit_year   = self.limit_year

        if crack_count >= CRACK_DENGER or rust_ratio >= RUST_DENGER:
            judgment      = "要点検"
            judgment_color = colors.HexColor("#cc3300")
        elif crack_count >= CRACK_WARN or rust_ratio >= RUST_WARN:
            judgment      = "経過観察"
            judgment_color = colors.HexColor("#e07000")
        else:
            judgment      = "異常なし"
            judgment_color = colors.HexColor("#1a7a3a")

        now = datetime.datetime.now()

        # コンテンツ構築
        story = []

        # タイトル
        story.append(Paragraph("コンクリート劣化診断レポート", style_title))
        story.append(Paragraph(
            f"作成日時: {now.strftime('%Y年%m月%d日  %H:%M:%S')}　　"
            f"使用モデル: {os.path.basename(MODEL_PATH)}",
            style_subtitle
        ))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
        story.append(Spacer(1, 4*mm))

        # 総合判定
        story.append(Paragraph("総合判定", style_section))
        judgment_table = Table(
            [[Paragraph(judgment, ParagraphStyle(
                "j", fontName=jp_font, fontSize=20, textColor=judgment_color, leading=26
            ))]],
            colWidths=[170*mm],
        )
        judgment_table.setStyle(TableStyle([
            ("BOX",       (0,0), (-1,-1), 1.5, judgment_color),
            ("BACKGROUND",(0,0), (-1,-1), colors.HexColor("#f9f9f9")),
            ("ALIGN",     (0,0), (-1,-1), "CENTER"),
            ("TOPPADDING",(0,0), (-1,-1), 6),
            ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ]))
        story.append(judgment_table)
        story.append(Spacer(1, 5*mm))

        # 解析結果数値
        story.append(Paragraph("解析結果", style_section))
        data = [
            ["項目", "計測値", "判定基準"],
            ["ひび割れ (crack)",  f"{crack_count} 個",  f"要注意: {CRACK_DENGER}個以上"],
            ["補修跡 (repair)",   f"{repair_count} 個",  "参考値"],
            ["目地 (seam)",       f"{seam_count} 個",    "参考値"],
            ["検出総数",          f"{total_count} 個",   ""],
            ["錆 占有率",         f"{rust_ratio:.1f} %",  f"要注意: {RUST_DENGER}%以上"],
            ["苔 占有率",         f"{moss_ratio:.1f} %",  "参考値"],
            ["推定寿命",          f"{limit_year} 年",     "中性化深さ計算より"],
        ]
        tbl = Table(data, colWidths=[55*mm, 55*mm, 60*mm])
        tbl.setStyle(TableStyle([
            ("FONTNAME",    (0,0), (-1,-1), jp_font),
            ("FONTSIZE",    (0,0), (-1,-1), 10),
            ("BACKGROUND",  (0,0), (-1, 0), colors.HexColor("#1a6fbc")),
            ("TEXTCOLOR",   (0,0), (-1, 0), colors.white),
            ("ALIGN",       (0,0), (-1,-1), "CENTER"),
            ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f0f4fa")]),
            ("BOX",         (0,0), (-1,-1), 0.5, colors.HexColor("#aaaaaa")),
            ("INNERGRID",   (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
            ("TOPPADDING",  (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 5*mm))

        # 解析画像
        story.append(Paragraph("AI解析画像（ひび割れ検出結果）", style_section))
        tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_img.close()
        cv2.imwrite(tmp_img.name, self._last_annotated)
        img_w = 170*mm
        orig_h, orig_w = self._last_annotated.shape[:2]
        img_h = img_w * (orig_h / orig_w)
        story.append(RLImage(tmp_img.name, width=img_w, height=img_h))
        story.append(Spacer(1, 4*mm))

        # 所見欄
        story.append(Paragraph("所見・備考", style_section))
        story.append(Paragraph(
            "（この欄は印刷後に手書きで記入してください）", style_body
        ))
        story.append(Spacer(1, 20*mm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(
            "※ 本レポートはAIによる自動解析結果であり、最終判断は有資格者による現地確認が必要です。",
            ParagraphStyle("note", fontName=jp_font, fontSize=8,
                           textColor=colors.HexColor("#888888"))
        ))

        doc.build(story)
        os.unlink(tmp_img.name)
    
    #ユーティリティ:ログ追記
    def _log(self,msg,level="ok"):
        prefix={"ok":"✓","warn":"⚠","err":"×"}.get(level," ")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END,prefix + msg + "\n",level)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.status_var.set(msg)

#エントリーポイント
if __name__ =="__main__":
    root=tk.Tk()
    app=ConcreteAnalyzerApp(root)
    root.mainloop()