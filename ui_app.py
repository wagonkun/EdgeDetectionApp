import cv2
from tkinter import *
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from edge_algorithms import apply_canny, apply_sobel, apply_laplacian

class EdgeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Edge Detection Studio")
        self.root.geometry("1300x750")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(True, True)

        self.image = None
        self.output = None
        self.current_algo = StringVar(value="Canny")

        self.build_layout()
        self.build_controls()

    #layout
    def build_layout(self):
        self.main_frame = Frame(self.root, bg="#1e1e1e")
        self.main_frame.pack(fill=BOTH, expand=True)

        #left: Image preview (side-by-side inside sub-frame)
        self.image_frame = Frame(self.main_frame, bg="#2a2a2a", width=850, height=700)
        self.image_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)

        #subframes for Input & Output
        self.left_panel = Frame(self.image_frame, bg="#2a2a2a")
        self.left_panel.pack(side=LEFT, expand=True, padx=10, pady=5)

        self.right_panel = Frame(self.image_frame, bg="#2a2a2a")
        self.right_panel.pack(side=RIGHT, expand=True, padx=10, pady=5)

        #labels above images
        Label(self.left_panel, text="Input Image", font=("Segoe UI", 12, "bold"),
              bg="#2a2a2a", fg="white").pack(pady=(5, 0))
        self.img_label_in = Label(self.left_panel, bg="#2a2a2a")
        self.img_label_in.pack(pady=5)

        Label(self.right_panel, text="Output Image", font=("Segoe UI", 12, "bold"),
              bg="#2a2a2a", fg="white").pack(pady=(5, 0))
        self.img_label_out = Label(self.right_panel, bg="#2a2a2a")
        self.img_label_out.pack(pady=5)

        #right: Control panel (scrollable)
        self.control_canvas = Canvas(self.main_frame, bg="#1e1e1e", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient=VERTICAL, command=self.control_canvas.yview)
        self.scroll_frame = Frame(self.control_canvas, bg="#1e1e1e")

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        )
        self.control_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.control_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.control_canvas.pack(side=LEFT, fill=BOTH, expand=False)
        self.scrollbar.pack(side=RIGHT, fill=Y)

    #controls
    def build_controls(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", background="#3a3a3a", foreground="white", font=("Segoe UI", 10))
        style.configure("TLabel", background="#1e1e1e", foreground="white", font=("Segoe UI", 10))
        style.configure("TScale", background="#1e1e1e")

        ttk.Button(self.scroll_frame, text="Upload Image", command=self.upload_image).pack(pady=10)
        ttk.Label(self.scroll_frame, text="Algorithm:").pack(pady=(20, 0))
        algo_menu = ttk.OptionMenu(self.scroll_frame, self.current_algo, "Canny", "Canny", "Sobel", "Laplacian", command=self.update_parameters)
        algo_menu.pack(pady=5)

        self.params_frame = Frame(self.scroll_frame, bg="#1e1e1e")
        self.params_frame.pack(pady=10, fill=X)
        self.update_parameters("Canny")

    #uploading image
    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            return
        self.image = img
        self.output = img.copy()
        self.display_images()
        self.update_output()

    #img display
    def display_images(self):
        if self.image is None:
            return

        def resize_for_display(img):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape
            max_w, max_h = 400, 400
            scale = min(max_w / w, max_h / h)
            return cv2.resize(img_rgb, (int(w * scale), int(h * scale)))

        img_in_disp = resize_for_display(self.image)
        img_out_disp = resize_for_display(self.output) if self.output is not None else img_in_disp

        imgtk_in = ImageTk.PhotoImage(Image.fromarray(img_in_disp))
        imgtk_out = ImageTk.PhotoImage(Image.fromarray(img_out_disp))

        self.img_label_in.imgtk = imgtk_in
        self.img_label_out.imgtk = imgtk_out
        self.img_label_in.config(image=imgtk_in)
        self.img_label_out.config(image=imgtk_out)

    #parameter handling
    def clear_params(self):
        for widget in self.params_frame.winfo_children():
            widget.destroy()

    def update_parameters(self, algo):
        self.clear_params()
        if algo == "Canny":
            self.create_slider("Lower Threshold", 0, 255, 50)
            self.create_slider("Upper Threshold", 0, 255, 150)
            self.create_slider("Kernel Size", 1, 15, 3, step=2)
            self.create_slider("Sigma", 0.1, 5.0, 1.0, res=0.1)
        elif algo == "Sobel":
            self.create_slider("Kernel Size", 1, 15, 3, step=2)
            self.create_dropdown("Direction", ["x", "y", "both"])
        elif algo == "Laplacian":
            self.create_slider("Kernel Size", 1, 15, 3, step=2)
        self.update_output()

    def create_slider(self, name, frm, to, init, step=1, res=None):
        ttk.Label(self.params_frame, text=name).pack()
        var = DoubleVar(value=init)
        scale = ttk.Scale(self.params_frame, from_=frm, to=to, orient=HORIZONTAL, variable=var)
        scale.pack(fill=X, padx=10, pady=5)
        value_lbl = Label(self.params_frame, text=f"{init:.2f}", bg="#1e1e1e", fg="#00ffaa")
        value_lbl.pack()
        scale.bind("<Motion>", lambda e: self.update_label(value_lbl, var))
        scale.bind("<ButtonRelease-1>", lambda e: self.update_output())
        setattr(self, f"{name.replace(' ', '_').lower()}_var", var)

    def create_dropdown(self, name, options):
        ttk.Label(self.params_frame, text=name).pack(pady=(10, 0))
        var = StringVar(value=options[0])
        opt = ttk.OptionMenu(self.params_frame, var, options[0], *options, command=lambda _: self.update_output())
        opt.pack(pady=5)
        setattr(self, f"{name.lower()}_var", var)

    def update_label(self, lbl, var):
        lbl.config(text=f"{var.get():.2f}")

    #processing
    def update_output(self):
        if self.image is None:
            return

        algo = self.current_algo.get()
        img = self.image.copy()

        if algo == "Canny":
            out = apply_canny(
                img,
                int(self.lower_threshold_var.get()),
                int(self.upper_threshold_var.get()),
                int(self.kernel_size_var.get()) | 1,
                float(self.sigma_var.get())
            )
        elif algo == "Sobel":
            out = apply_sobel(
                img,
                int(self.kernel_size_var.get()) | 1,
                self.direction_var.get()
            )
        else:
            out = apply_laplacian(img, int(self.kernel_size_var.get()) | 1)

        out_rgb = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        self.output = out_rgb
        self.display_images()
