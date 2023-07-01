import json
import math
import tkinter
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from tkinter.messagebox import showinfo
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.patches as plt_patches

from utils import SemanticFrames


class PredictVisualizer:
    text_style = dict(picker=True, size=8, color='k', zorder=10, ha="center")
    text_box_style = dict(boxstyle="round,pad=0.2", fc="yellow", alpha=.6, ec="black", lw=0.2)

    # Colors
    CAR = '#F1C40F'
    EGO = '#AB4000'
    CAR_OUTLINE = '#B7950B'
    CAR_FRONT = '#00F000'
    CAR_BACK = '#0000F0'

    def __init__(self, semantic_frames: list[SemanticFrames], gradient_fnc: tuple[Callable, Callable], traj):

        self.semantic_frames = semantic_frames

        self.template = None

        self._job = None

        self.templates = ["cut_in", "cut_out", "drift_in", "drift_out"]
        self.out = {"cut_in": {}, "cut_out": {}, "drift_in": {}, "drift_out": {}}
        self.selected = None
        self.added = False
        self.selected_fellow = None
        self.scatter = None
        self.x1 = self.x2 = None
        self.patches = {}
        self.annotations = {}
        self.lat_seg_alg = None

        self.root = tkinter.Tk()
        self.root.wm_title("Embedding in Tk")

        fig, ax = plt.subplots(figsize=(10, 2), dpi=100, nrows=1, ncols=1)
        fig.tight_layout()

        fig2, self.sp = plt.subplots(figsize=(15, 3), dpi=100, nrows=1, ncols=1)
        fig2.tight_layout()

        self.sp2 = ax


        self.canvas2 = FigureCanvasTkAgg(fig2, master=self.root)  # A tk.DrawingArea.
        self.canvas2.draw()
        self.canvas2.get_tk_widget().grid(column=0, row=1)

        # self.toolbar = NavigationToolbar2Tk(self.canvas, self.root, pack_toolbar=False)
        # self.toolbar.update()

        # create a list box
        langs_var = tkinter.StringVar(value="")

        self.listbox = tkinter.Listbox(
            self.root,
            listvariable=langs_var,
            height=20,
            selectmode='extended')

        self.listbox.grid(column=1, row=1)

        self.scrollbar = tkinter.Scrollbar(self.root)
        self.scrollbar.grid(column=2, row=1, sticky='ns')

        self.listbox.config(yscrollcommand=self.scrollbar.set)

        self.scrollbar.config(command=self.listbox.yview)

        self.slider = tkinter.Scale(master=self.root, from_=0, to=len(semantic_frames)-1,
                                    orient="horizontal",
                                    command=self.updateValue,
                                    length=1000)
        self.slider.grid(row=2)


        self.root.bind('<Key>', self.on_key_press)

        self.w = tkinter.Label(self.root, text="Hello, Tk!")
        self.w.grid(column=1, row=2)

        self.show_frames(self.semantic_frames[0])
        self.sp.set_xlim([-10, 60])
        self.sp.set_ylim([-10, 16])

        X, Y = np.meshgrid(np.linspace(-10, 60, 20), np.linspace(-10, 16, 20))

        U = [gradient_fnc[0](x1, y1) for x1, y1 in zip(X, Y)]
        V = [gradient_fnc[1](x1, y1) for x1, y1 in zip(X, Y)]

        self.sp.quiver(X, Y, U, V, linewidth=0.10)
        self.sp.plot(traj[0], traj[1])


        tkinter.mainloop()



    def _bind_elements(self):
        self.listbox.bind('<<ListboxSelect>>',
                          lambda event=None, attr="listbox": self.template.items_selected(event, attr))

    def on_key_press(self, event):
        print("you pressed {}".format(event.char))

        if str(event.keysym) == 'Left':
            self.slider.set(self.slider.get() - 1)
            self.slider.update()

        if str(event.keysym) == 'Right':
            self.slider.set(self.slider.get() + 1)
            self.slider.update()
    def updateValue(self, event):
        if self._job:
            self.root.after_cancel(self._job)
        self._job = self.root.after(20, self.update_plot)

    def update_plot(self):
        self._job = None
        frame = int(self.slider.get())
        self.show_frames(self.semantic_frames[frame])


    def _quit(self):
        self.root.quit()  # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent
        # Fatal Python Error: PyEval_RestoreThread: NULL tstate

    def reset(self):
        self.out_df_l = list()
        self.lane_lines = list()
        self.patches = {}
        self.annotations = {}


    def rectangle(self, x1, x2):
        self.x1 = round(x1)
        self.x2 = round(x2)

    def show_frames(self, semantic_frames: SemanticFrames) -> None:
        for i in self.patches.values():
            i.remove()

        for i in self.annotations.values():
            i.remove()

        self.annotations = {}
        self.patches = {}

        for frame in semantic_frames.values():
            cog = (frame.s, frame.d)
            yaw = frame.yaw
            length = frame.length
            width = frame.width
            car = plt_patches.Rectangle(cog, width=length, height=width,
                                        angle=np.rad2deg(yaw),
                                        facecolor=self.CAR if not frame.is_ego else self.EGO,
                                        edgecolor=self.CAR_OUTLINE, zorder=20)

            center_anno = (cog[0] - 2, cog[1] - 2)
            anno = self.sp.annotate(frame.object_id, xy=cog, xytext=center_anno,
                                        bbox=self.text_box_style, **self.text_style)


            # Add rectangle to current axis
            self.patches[frame.object_id] = self.sp.add_patch(car)
            self.annotations[frame.object_id] = anno

        self.canvas2.draw()




if __name__ == '__main__':
    a = App()
