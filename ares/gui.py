import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import pyglet
import matplotlib as mpl
import csv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import style
import matplotlib.font_manager as fm
import downtime, kpicalc, mle, modsel, pred, simulation
mpl.use("TkAgg")
style.use("ggplot")


pyglet.font.add_file('./fonts/TecnicoFino.ttf')
prop = fm.FontProperties(fname='./fonts/TecnicoFino.ttf')


LARGE_FONT = ("Tecnico Fino", 12)
NORM_FONT = ("Tecnico Fino", 10)
SMALL_FONT = ("Tecnico Fino", 8)

f = Figure()
a = f.add_subplot(111)

model = "WPLP"
gamma = 1.2
alpha = .04
beta = .85

nttf = (1000, "hours")

datax = [1, 2, 3, 4, 5, 6, 7, 8]
datay = [1, 1, 2, 3, 5, 8, 13, 21]

project_name = "Analysis 1"

def analysis():
    global failure_data
    global res_failure_model
    global gamma, alpha, beta
    global T
    T = list()
    for i in range(len(failure_data)):
        if i == 0:
            T.append(failure_data[0])
        else:
            T.append(failure_data[i] + T[-1])
    res_failure_model = modsel.akaike_information_criterion(mle.failure_models(T))
    kpicalc.model_parameters(res_failure_model, disp=False)

def print_debug():
    global failure_data
    try:
        print(failure_data)
    except NameError:
        pass


def import_data(self):
    global failure_data
    self.filename = filedialog.askopenfilename(initialdir="./", title="Select a file",
                                               filetypes=(("CSV files", "*.csv"), ("All Files", "*.*")))
    with open(self.filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i in csv_reader:
            failure_data = i
    failure_data = [float(i) for i in failure_data]


def popupmsg(msg):
    # showinfo, showwarning, showerror, askquestion, askokcancel, askyesno
    messagebox.showwarning("Error", msg)


class AresApp(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self, default="./images/icon.ico")
        tk.Tk.wm_title(self, "ares")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New Project...", command=lambda: popupmsg("Not implemented."))
        filemenu.add_command(label="Load Project...", command=lambda: popupmsg("Not implemented."))
        filemenu.add_command(label="Save Project...", command=lambda: popupmsg("Not implemented."))
        filemenu.add_separator()
        filemenu.add_command(label="Export Results", command=lambda: popupmsg("Not implemented."))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=quit)
        menubar.add_cascade(label="File", menu=filemenu)

        tk.Tk.config(self, menu=menubar)

        self.frames = {}

        for F in (StartPage, ResultsPage):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text=project_name + ": Set-up", font=LARGE_FONT)
        label.grid(row=0, column=0, columns=10)

        separator = ttk.Separator(self, orient=tk.HORIZONTAL)
        separator.grid(row=1, column=0, sticky="ew", columns=10)

        gen_frame = tk.LabelFrame(self, font=NORM_FONT)
        gen_frame.grid(row=2, column=0, sticky="nsew")

        def RadioBtnSelected():
            print(rb_model.get())

        rb_model = tk.StringVar()

        rb_models = [
            ("Default", "default"),
            ("Weibull-Power-Law Process", "WPLP"),
            ("Nonhomogeneous Poisson Process", "NHPP"),
            ("Homogeneous Poisson Process", "HPP"),
            ("Weibull Renewal Process", "WRP")
        ]

        fail_model_sel_frame = tk.LabelFrame(gen_frame, text="Failure Model Selection", font=NORM_FONT)
        fail_model_sel_frame.grid(row=0, column=0, padx=5, pady=5)

        rb_model = tk.StringVar()
        rb_model.set("default")

        rb_Def = tk.Radiobutton(fail_model_sel_frame, text=rb_models[0][0], variable=rb_model, font=NORM_FONT,
                                value=rb_models[0][1], command=RadioBtnSelected)
        rb_WPLP = tk.Radiobutton(fail_model_sel_frame, text=rb_models[1][0], variable=rb_model, font=NORM_FONT,
                                 value=rb_models[1][1], state=tk.DISABLED, command=RadioBtnSelected)
        rb_NHPP = tk.Radiobutton(fail_model_sel_frame, text=rb_models[2][0], variable=rb_model, font=NORM_FONT,
                                 value=rb_models[2][1], state=tk.DISABLED, command=RadioBtnSelected)
        rb_HPP = tk.Radiobutton(fail_model_sel_frame, text=rb_models[3][0], variable=rb_model, font=NORM_FONT,
                                value=rb_models[3][1], state=tk.DISABLED, command=RadioBtnSelected)
        rb_WRP = tk.Radiobutton(fail_model_sel_frame, text=rb_models[4][0], variable=rb_model, font=NORM_FONT,
                                value=rb_models[4][1], state=tk.DISABLED, command=RadioBtnSelected)

        rb_Def.grid(row=0, column=0, sticky="w")
        rb_WPLP.grid(row=1, column=0, sticky="w")
        rb_NHPP.grid(row=2, column=0, sticky="w")
        rb_HPP.grid(row=3, column=0, sticky="w")
        rb_WRP.grid(row=4, column=0, sticky="w")

        import_frame = tk.LabelFrame(gen_frame)
        import_frame.grid(row=0, column=1)

        btn_import_failure_data = tk.Button(import_frame, text="Import failure data...", font=NORM_FONT,
                                            command=lambda: import_data(AresApp))
        btn_import_downtime_data = tk.Button(import_frame, text="Import downtime data...", font=NORM_FONT)

        btn_import_failure_data.grid(row=0, column=0, padx=5, pady=5)
        btn_import_downtime_data.grid(row=1, column=0, padx=5, pady=5)

        # btn_test = tk.Button(self, text="Run analysis", font=NORM_FONT,
        #                      command=print_debug)
        # btn_test.place(relx=.5, rely=.5, anchor="se")

        button = tk.Button(self, text="Run analysis", font=NORM_FONT,
                           command=lambda: controller.show_frame(ResultsPage))
        button.place(relx=.95, rely=.95, anchor="se")

        self.columnconfigure(0, weight=1)


class ResultsPage(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text=project_name + ": Results", font=LARGE_FONT)
        label.grid(row=0, column=0, columns=10)

        separator = ttk.Separator(self, orient=tk.HORIZONTAL)
        separator.grid(row=1, column=0, sticky="ew", columns=10)

        tech_info_frame = tk.LabelFrame(self, borderwidth=0)
        tech_info_frame.grid(row=2, column=0, sticky="new")

        model_info_frame = tk.LabelFrame(tech_info_frame, text="Model Information", font=NORM_FONT, padx=10, pady=10)
        model_info_frame.grid(row=0, column=0, padx=5, pady=5, sticky="new")

        ttk.Label(model_info_frame, text="Model:", font=(*NORM_FONT, "bold")).grid(row=0, column=0, sticky='e')
        ttk.Label(model_info_frame, text=u"\u03b3:", font=(*NORM_FONT, "bold")).grid(row=1, column=0, sticky='e')
        ttk.Label(model_info_frame, text=u"\u03b1:", font=(*NORM_FONT, "bold")).grid(row=2, column=0, sticky='e')
        ttk.Label(model_info_frame, text=u"\u03b2:", font=(*NORM_FONT, "bold")).grid(row=3, column=0, sticky='e')

        ttk.Label(model_info_frame, text=model, font=NORM_FONT).grid(row=0, column=1, sticky='w')
        ttk.Label(model_info_frame, text=gamma, font=NORM_FONT).grid(row=1, column=1, sticky='w')
        ttk.Label(model_info_frame, text=alpha, font=NORM_FONT).grid(row=2, column=1, sticky='w')
        ttk.Label(model_info_frame, text=beta, font=NORM_FONT).grid(row=3, column=1, sticky='w')

        pred_frame = tk.LabelFrame(tech_info_frame, text="Prediction", font=NORM_FONT, padx=10, pady=10)
        pred_frame.grid(row=2, column=0, padx=5, pady=5, sticky="new")

        pred_ttf = tk.Text(pred_frame, width=4, height=1, borderwidth=0,
                           background=self.cget("background"), font=(*NORM_FONT, "bold"))
        pred_ttf.tag_configure("subscript", offset=-4)
        pred_ttf.insert("insert", "T", "", "n+1", "subscript", ":")
        pred_ttf.configure(state="disabled")
        pred_ttf.grid(row=0, column=0, sticky='e')
        ttk.Label(pred_frame, text=nttf, font=NORM_FONT).grid(row=0, column=1)

        plot_frame = tk.LabelFrame(self, borderwidth=0)
        plot_frame.grid(row=2, column=1, sticky="nsew")

        a.scatter(datax, datay, label="Fibonacci")

        a.legend(bbox_to_anchor=(0, 1.02, 1, .102), loc=3, ncol=1, borderaxespad=0)

        title = "Fibonacci Numbers\nLast value: " + str(datay[-1])
        x_axis = "n"
        y_axis = "Fibonacci number"
        a.set_title(title, fontproperties=prop, size=LARGE_FONT[1])
        a.set_xlabel(x_axis, fontproperties=prop, size=NORM_FONT[1])
        a.set_ylabel(y_axis, fontproperties=prop, size=NORM_FONT[1])

        canvas = FigureCanvasTkAgg(f, plot_frame)
        canvas.draw()
        # canvas.get_tk_widget().grid(row=2, column=2, rowspan=2)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        gen_report = tk.Button(tech_info_frame, text="Generate Results Report", font=NORM_FONT,
                               command=lambda: popupmsg("Not implemented."))
        gen_report.grid(row=3, column=0, sticky='nsew', padx=5, pady=5)

        back = tk.Button(tech_info_frame, text="Return", font=NORM_FONT,
                         command=lambda: controller.show_frame(StartPage))
        back.grid(row=4, column=0, sticky='nsew', padx=5, pady=5)


app = AresApp()

# w, h = app.winfo_screenwidth(), app.winfo_screenheight()
# app.geometry("%dx%d+0+0" % (w, h))
app.state("zoomed")
app.mainloop()
