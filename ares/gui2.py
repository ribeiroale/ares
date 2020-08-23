import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import csv
import pyglet
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import style
import matplotlib.font_manager as fm
import downtime, kpicalc, mle, modsel, pred, simulation
from scipy.integrate import simps
from numpy import linspace
from decimal import Decimal

mpl.use("TkAgg")
mpl.rcParams['savefig.dpi'] = 300
style.use("ggplot")


pyglet.font.add_file('./fonts/TecnicoFino.ttf')
prop = fm.FontProperties(fname='./fonts/TecnicoFino.ttf')

LARGE_FONT = ("Tecnico Fino", 12)
NORM_FONT = ("Tecnico Fino", 10)
SMALL_FONT = ("Tecnico Fino", 8)

project_name = "Belinelli Data (2015)"

root = tk.Tk()
root.iconbitmap(default="./images/icon.ico")
root.wm_title("ares")

# failure_data, downtime_data = list(), list()
res_failure_model, res_downtime_model = list(), list()
gamma, alpha, beta = 1, 1, 1
availability, predictions = list(), list()


class Tooltip:
    '''
    It creates a tooltip for a given widget as the mouse goes on it.

    see:

    http://stackoverflow.com/questions/3221956/
           what-is-the-simplest-way-to-make-tooltips-
           in-tkinter/36221216#36221216

    http://www.daniweb.com/programming/software-development/
           code/484591/a-tooltip-class-for-tkinter

    - Originally written by vegaseat on 2014.09.09.

    - Modified to include a delay time by Victor Zaccardo on 2016.03.25.

    - Modified
        - to correct extreme right and extreme bottom behavior,
        - to stay inside the screen whenever the tooltip might go out on
          the top but still the screen is higher than the tooltip,
        - to use the more flexible mouse positioning,
        - to add customizable background color, padding, waittime and
          wraplength on creation
      by Alberto Vassena on 2016.11.05.

      Tested on Ubuntu 16.04/16.10, running Python 3.5.2

    TODO: themes styles support
    '''

    def __init__(self, widget,
                 *,
                 bg='#FFFFEA',
                 pad=(5, 3, 5, 3),
                 text='widget info',
                 waittime=400,
                 wraplength=250):

        self.waittime = waittime  # in miliseconds, originally 500
        self.wraplength = wraplength  # in pixels, originally 180
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.onEnter)
        self.widget.bind("<Leave>", self.onLeave)
        self.widget.bind("<ButtonPress>", self.onLeave)
        self.bg = bg
        self.pad = pad
        self.id = None
        self.tw = None

    def onEnter(self, event=None):
        self.schedule()

    def onLeave(self, event=None):
        self.unschedule()
        self.hide()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.show)

    def unschedule(self):
        id_ = self.id
        self.id = None
        if id_:
            self.widget.after_cancel(id_)

    def show(self):
        def tip_pos_calculator(widget, label,
                               *,
                               tip_delta=(10, 5), pad=(5, 3, 5, 3)):

            w = widget

            s_width, s_height = w.winfo_screenwidth(), w.winfo_screenheight()

            width, height = (pad[0] + label.winfo_reqwidth() + pad[2],
                             pad[1] + label.winfo_reqheight() + pad[3])

            mouse_x, mouse_y = w.winfo_pointerxy()

            x1, y1 = mouse_x + tip_delta[0], mouse_y + tip_delta[1]
            x2, y2 = x1 + width, y1 + height

            x_delta = x2 - s_width
            if x_delta < 0:
                x_delta = 0
            y_delta = y2 - s_height
            if y_delta < 0:
                y_delta = 0

            offscreen = (x_delta, y_delta) != (0, 0)

            if offscreen:

                if x_delta:
                    x1 = mouse_x - tip_delta[0] - width

                if y_delta:
                    y1 = mouse_y - tip_delta[1] - height

            offscreen_again = y1 < 0  # out on the top

            if offscreen_again:
                # No further checks will be done.

                # TIP:
                # A further mod might automagically augment the
                # wraplength when the tooltip is too high to be
                # kept inside the screen.
                y1 = 0

            return x1, y1

        bg = self.bg
        pad = self.pad
        widget = self.widget

        # creates a toplevel window
        self.tw = tk.Toplevel(widget)

        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)

        win = tk.Frame(self.tw,
                       background=bg,
                       borderwidth=0)
        label = tk.Label(win,
                          text=self.text,
                          justify=tk.LEFT,
                          background=bg,
                          relief=tk.SOLID,
                          borderwidth=0,
                          wraplength=self.wraplength)

        label.grid(padx=(pad[0], pad[2]),
                   pady=(pad[1], pad[3]),
                   sticky=tk.NSEW)
        win.grid()

        x, y = tip_pos_calculator(widget, label)

        self.tw.wm_geometry("+%d+%d" % (x, y))

    def hide(self):
        tw = self.tw
        if tw:
            tw.destroy()
        self.tw = None


def run_analysis(data_failure, data_downtime, rep=100):

    global res_failure_model
    global res_downtime_model
    global gamma
    global alpha
    global beta
    global availability
    global predictions
    global kaplan_meier_obj
    global renewal_prop
    global num_failures
    global gammaLower, gammaUpper, alphaLower, alphaUpper, betaLower, betaUpper
    global conf_int_A

    T = list()
    for i in range(len(data_failure)):
        if i == 0:
            T.append(data_failure[0])
        else:
            T.append(data_failure[i] + T[-1])

    res_failure_model = modsel.akaike_information_criterion(mle.failure_models(T))
    # res_failure_model = mle.failure_models(T)[1]
    res_downtime_model = downtime.downtime_accepted_models(data_downtime)[1]
    gamma, alpha, beta = kpicalc.model_parameters(res_failure_model, disp=False)[0]
    ci_params = kpicalc.model_parameters(res_failure_model, disp=False)[1]

    gammaLower, gammaUpper = ci_params[0]
    alphaLower, alphaUpper = ci_params[1]
    betaLower, betaUpper = ci_params[2]

    T_G, S = simulation.failure_repair_process(data_failure, data_downtime)
    S = [0] + S

    T_G_R, S_R, X_R, D_R, T_R = simulation.sim(rep, res_failure_model, res_downtime_model,
                                               numfail=len(T), timeHorizon=max(T_G))

    # t = kpicalc.time_axis(1, T_G)
    t = linspace(0, max(T_G)).tolist()
    A, conf_int_A = kpicalc.A_Sim(t, T_G_R, S_R)
    availability = (t, A)

    predictions = [pred.point_predictor(T[-1], res_failure_model), pred.interval_predictor(T[-1], res_failure_model)]
    kaplan_meier_obj = [kpicalc.reliability(n, len(data_failure)+1, X_R) for n in range(1, len(data_failure)+2)]
    renewal_prop = kpicalc.rs_method(500, T, res_failure_model)
    num_failures = kpicalc.num_failures(T)

    btn_results["state"] = "normal"


def show_results():

    results_win = tk.Frame(root)
    results_win.grid(row=3, column=0, sticky="nsew")

    model = res_failure_model[0]

    tech_info_frame = tk.Frame(results_win)
    tech_info_frame.grid(row=0, column=0, sticky="nsew")

    model_info_frame = tk.LabelFrame(tech_info_frame, text="Model Information", font=NORM_FONT, padx=10, pady=10)
    model_info_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

    # ttk.Label(model_info_frame, text="Model:", font=(*NORM_FONT, "bold")).grid(row=0, column=0, sticky='e')
    # ttk.Label(model_info_frame, text=u"\u03b1:", font=(*NORM_FONT, "bold")).grid(row=1, column=0, sticky='e')
    # ttk.Label(model_info_frame, text="a:", font=(*NORM_FONT, "bold")).grid(row=2, column=0, sticky='e')
    # ttk.Label(model_info_frame, text="b:", font=(*NORM_FONT, "bold")).grid(row=3, column=0, sticky='e')

    label_1 = ttk.Label(model_info_frame, text="Model:", font=(*NORM_FONT, "bold"))
    label_2 = ttk.Label(model_info_frame, text=u"\u03b1:", font=(*NORM_FONT, "bold"))
    label_3 = ttk.Label(model_info_frame, text="a:", font=(*NORM_FONT, "bold"))
    label_4 = ttk.Label(model_info_frame, text="b:", font=(*NORM_FONT, "bold"))

    label_1.grid(row=0, column=0, sticky='e')
    label_2.grid(row=1, column=0, sticky='e')
    label_3.grid(row=2, column=0, sticky='e')
    label_4.grid(row=3, column=0, sticky='e')

    ttk.Label(model_info_frame, text=model, font=NORM_FONT).grid(row=0, column=1, sticky='w')
    ttk.Label(model_info_frame, text=f"{gammaLower:.3f} < {gamma:.3f} < {gammaUpper:.3f}", font=NORM_FONT).grid(row=1, column=1, sticky='w')
    ttk.Label(model_info_frame, text=f"{alphaLower:.3f} < {alpha:.3f} < {alphaUpper:.3f}", font=NORM_FONT).grid(row=2, column=1, sticky='w')
    ttk.Label(model_info_frame, text=f"{betaLower:.3f} < {beta:.3f} < {betaUpper:.3f}", font=NORM_FONT).grid(row=3, column=1, sticky='w')

    pred_frame = tk.LabelFrame(tech_info_frame, text="Prediction", font=NORM_FONT, padx=10, pady=10)
    pred_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

    pred_ttf = tk.Text(pred_frame, width=5, height=2, borderwidth=0,
                       background=results_win.cget("background"), font=(*NORM_FONT, "bold"))
    pred_ttf.tag_configure("subscript", offset=-4)
    pred_ttf.insert("insert", "T", "", "n+1", "subscript", ":")
    pred_ttf.configure(state="disabled")
    pred_ttf.grid(row=0, column=0, sticky='e')

    pred_tbf = tk.Text(pred_frame, width=5, height=2, borderwidth=0,
                       background=results_win.cget("background"), font=(*NORM_FONT, "bold"))
    pred_tbf.tag_configure("subscript", offset=-4)
    pred_tbf.insert("insert", "X", "", "n+1", "subscript", ":")
    pred_tbf.configure(state="disabled")
    pred_tbf.grid(row=1, column=0, sticky='e')

    sep1 = ttk.Separator(pred_frame, orient=tk.HORIZONTAL)
    sep1.grid(row=2, column=0, columnspan=3, sticky="ew")

    pred_ttf_lower = tk.Text(pred_frame, width=6, height=2, borderwidth=0,
                             background=results_win.cget("background"), font=(*NORM_FONT, "bold"))
    pred_ttf_lower.tag_configure("subscript", offset=-4)
    pred_ttf_lower.tag_configure("superscript", offset=+4)
    pred_ttf_lower.insert("insert", "T", "", "n+1", "subscript", "L", "superscript", ":")
    pred_ttf_lower.configure(state="disabled")
    pred_ttf_lower.grid(row=3, column=0, sticky='e')

    pred_ttf_upper = tk.Text(pred_frame, width=6, height=2, borderwidth=0,
                             background=results_win.cget("background"), font=(*NORM_FONT, "bold"))
    pred_ttf_upper.tag_configure("subscript", offset=-4)
    pred_ttf_upper.tag_configure("superscript", offset=+4)
    pred_ttf_upper.insert("insert", "T", "", "n+1", "subscript", "U", "superscript", ":")
    pred_ttf_upper.configure(state="disabled")
    pred_ttf_upper.grid(row=4, column=0, sticky='e')

    ttk.Label(pred_frame, text=f"{predictions[0][0]:.3f}" + " hours", font=NORM_FONT).grid(row=0, column=1)
    ttk.Label(pred_frame, text=f"{predictions[0][1]:.3f}" + " hours", font=NORM_FONT).grid(row=1, column=1)
    ttk.Label(pred_frame, text=f"{predictions[1][0]:.3f}" + " hours", font=NORM_FONT).grid(row=3, column=1)
    ttk.Label(pred_frame, text=f"{predictions[1][1]:.3f}" + " hours", font=NORM_FONT).grid(row=4, column=1)

    plot_frame = tk.Frame(results_win)
    plot_frame.grid(row=0, column=1, sticky="nsew")

    f = Figure()
    a1 = f.add_subplot(221)

    f.subplots_adjust(hspace=.8)

    a1.plot(availability[0][1:], availability[1][1:], label="Availability")
    a1.fill_between(availability[0][1:], (availability[1][1:]-conf_int_A[1:]), (availability[1][1:]+conf_int_A[1:]),
                    alpha=.2)
    a1.legend(bbox_to_anchor=(0, 1.02, 1, .102), loc=3, ncol=1, borderaxespad=0)

    #title = "Availability\nLast value: " + f"{availability[1][-1]: .4f}"
    x_axis = "Global Time"
    y_axis = "Availability"
    # a.set_title(title, fontproperties=prop, size=LARGE_FONT[1])
    a1.set_xlabel(x_axis, fontproperties=prop, size=NORM_FONT[1])
    a1.set_ylabel(y_axis, fontproperties=prop, size=NORM_FONT[1])
    text_str1 = f"Last value: {availability[1][-1]:.4f} ± {Decimal(str(conf_int_A[-1])):.2E}"
    # text_str1 = 'Last value: %.4f ± %.4f' % (availability[1][-1], conf_int_A[-1])
    box_prop = dict(boxstyle='round', alpha=0.5, facecolor='wheat')
    a1.text(.9, .2, text_str1, transform=a1.transAxes, fontsize=NORM_FONT[1],
            verticalalignment='bottom', horizontalalignment='right', bbox=box_prop)

    a2 = f.add_subplot(223)

    x2 = kaplan_meier_obj[-1].timeline
    y2 = [kaplan_meier_obj[-1].survival_function_.values[i][0] for i in range(0, len(kaplan_meier_obj[-1].survival_function_.values))]
    median2 = kaplan_meier_obj[-1].median_survival_time_
    mean2 = simps(y2, x2)
    conf_interv = kaplan_meier_obj[-1].confidence_interval_survival_function_
    lower_ci = [conf_interv.values[i][0] for i in range(0, len(conf_interv))]
    upper_ci = [conf_interv.values[i][1] for i in range(0, len(conf_interv))]

    a2.step(x2, y2, label='Kaplan Meier Estimate for the next TBF')
    a2.fill_between(x2, lower_ci, upper_ci, alpha=.2, step='pre')

    #a2.step(x2, lower_ci, label='Lower 95% confidence interval')
    #a2.step(x2, upper_ci, label='Upper 95% confidence interval')
    a2.legend(bbox_to_anchor=(0, 1.02, 1, .102), loc=3, ncol=1, borderaxespad=0)

    # title2 = f"Reliability function of the {len(kaplan_meier_obj)-1}th time-between-failure"
    x_axis2 = "Time"
    y_axis2 = "Reliability"
    # a2.set_title(title2, fontproperties=prop, size=LARGE_FONT[1])
    a2.set_xlabel(x_axis2, fontproperties=prop, size=NORM_FONT[1])
    a2.set_ylabel(y_axis2, fontproperties=prop, size=NORM_FONT[1])
    text_str2 = '\n'.join(('mean = %.2f' % (mean2,), 'median = %.2f' % (median2,)))
    box_prop = dict(boxstyle='round', alpha=0.5, facecolor='wheat')
    a2.text(.9, .9, text_str2, transform=a2.transAxes, fontsize=NORM_FONT[1],
            verticalalignment='top', horizontalalignment='right', bbox=box_prop)

    a3 = f.add_subplot(222)

    x3_1 = renewal_prop[0]
    y3_1 = renewal_prop[1][0]
    x3_2 = num_failures[0]
    y3_2 = num_failures[1]

    a3.plot(x3_1, y3_1, '.-', markersize=.2, label="Expected number of failures")
    a3.step(x3_2, y3_2, label="Observed number of failures")
    a3.legend(bbox_to_anchor=(0, 1.02, 1, .102), loc=3, ncol=1, borderaxespad=0)

    x_axis3 = "Time"
    y_axis3 = "Number of failures"
    a3.set_xlabel(x_axis3, fontproperties=prop, size=NORM_FONT[1])
    a3.set_ylabel(y_axis3, fontproperties=prop, size=NORM_FONT[1])

    a4 = f.add_subplot(224)

    x4 = renewal_prop[0]
    y4 = renewal_prop[1][1]

    a4.plot(x4, y4, '.-', label="Rate of occurrence of failures")

    x_axis4 = "Time"
    y_axis4 = "Rate of occurrence of failures"
    a4.set_xlabel(x_axis4, fontproperties=prop, size=NORM_FONT[1])
    a4.set_ylabel(y_axis4, fontproperties=prop, size=NORM_FONT[1])

    canvas = FigureCanvasTkAgg(f, plot_frame)
    canvas.draw()
    # canvas.get_tk_widget().grid(row=2, column=2, rowspan=2)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    results_win.columnconfigure(1, weight=1)
    results_win.rowconfigure(0, weight=1)

    Tooltip(pred_ttf,
            text='Point predictor for the time to next failure in the original time scale', wraplength=200)
    Tooltip(pred_ttf_upper,
            text='Upper interval predictor for the time to next failure in the original time scale', wraplength=200)
    Tooltip(pred_ttf_lower,
            text='Lower interval predictor for the time to next failure in the original time scale', wraplength=200)
    Tooltip(pred_tbf,
            text='Point predictor for the next time between failures in the original time scale', wraplength=200)
    Tooltip(label_2,
            text='Renewal parameter', wraplength=200)
    Tooltip(label_3,
            text='Scale parameter', wraplength=200)
    Tooltip(label_4,
            text='Time dependence parameter', wraplength=200)

    gen_report = tk.Button(tech_info_frame, text="Generate Results Report", font=NORM_FONT,
                           command=lambda: pop_up_msg("Not implemented."))
    gen_report.grid(row=3, column=0, sticky='nsew', padx=5, pady=5)


def print_debug(data):
    try:
        print(data)
    except NameError:
        pass


def pop_up_msg(msg):
    # showinfo, showwarning, showerror, askquestion, askokcancel, askyesno
    messagebox.showwarning("Error", msg)


def import_data(self, origin=""):
    global failure_data
    global downtime_data
    self.filename = filedialog.askopenfilename(initialdir="./", title="Select a file",
                                               filetypes=(("CSV files", "*.csv"), ("All Files", "*.*")))
    with open(self.filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i in csv_reader:
            tmp_list = i
    if origin == "failure":
        failure_data = [float(i) for i in tmp_list]
        btn_import_downtime_data["state"] = "normal"
    elif origin == "downtime":
        downtime_data = [float(i) for i in tmp_list]
        btn_run_analysis["state"] = "normal"


label = ttk.Label(root, text=project_name, font=LARGE_FONT)
label.grid(row=0, column=0)

separator = ttk.Separator(root, orient=tk.HORIZONTAL)
separator.grid(row=1, column=0, sticky="ew")

set_up_frame = tk.Frame(root)
set_up_frame.grid(row=2, column=0, sticky="ew")

btn_import_failure_data = ttk.Button(set_up_frame, text="Import failure data...",
                                     command=lambda: import_data(root, origin="failure"))
btn_import_failure_data.grid(row=0, column=0)

btn_import_downtime_data = ttk.Button(set_up_frame, text="Import downtime data...", state=tk.DISABLED,
                                      command=lambda: import_data(root, origin="downtime"))
btn_import_downtime_data.grid(row=0, column=1)

btn_run_analysis = ttk.Button(set_up_frame, text="Run Analysis", state=tk.DISABLED,
                              command=lambda: run_analysis(failure_data, downtime_data))
btn_run_analysis.grid(row=0, column=2)

btn_results = ttk.Button(set_up_frame, text="Show Results", state=tk.DISABLED,
                         command=show_results)
btn_results.grid(row=0, column=3)

root.columnconfigure(0, weight=1)
root.rowconfigure(3, weight=1)

root.state("zoomed")
root.mainloop()
