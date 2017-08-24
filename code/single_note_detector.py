# -*- coding: utf-8 -*-
"""
Created on May 23 2014
# class taken from the SciPy 2015 Vispy talk opening example
# see https://github.com/vispy/vispy/pull/928
@author: florian
# librosa: https://librosa.github.io/librosa/generated/librosa.feature.chroma_cqt.html#librosa.feature.chroma_cqt
"""
import sys
import threading
import atexit
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time
import librosa
from PyQt4 import QtGui, uic, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

# variables in the program (not necessarily easily changed):
# chunksize - elaborated in class MicrophoneRecorder
# noise (for energy) - determines the onset detection desensitivity
# TODO: calculate percentage of area under graph due to the sound from ffreq
# threshold crossing point - determines the onset detection sensitivity
# lower threshold of spectrum - muting everything below the frequency

class MicrophoneRecorder(object):
    """
    This microphone runs on an independent thread.
    It groups the signal in blocks of 2048 (chunksize) entries, as "frame"
    It accumulate these frames until "get_frames" is called, when it will pass over these entries.
    (There should be no accumulation of entries, else information is lost.)
    Choice of chunksize
    - large enough to determine the exact frequency
    - small enough to be responsive: to indicate the new note as promptly as possible
    """
    def __init__(self, rate=44100, chunksize=2048):
        self.rate = rate  # sampling rate of microphone
        self.chunksize = chunksize  # size of each "frames"
        self.p = pyaudio.PyAudio()  # imported object to interface with the microphone
        self.stream = self.p.open(format=pyaudio.paInt16,  # sound take the format of int16
                                  channels=1,  # takes mono?
                                  rate=self.rate,  # sampling rate
                                  input=True,
                                  frames_per_buffer=self.chunksize,  # size of each "frame"
                                  stream_callback=self.new_frame)  # function to call per "frame" generated
        self.lock = threading.Lock()  # something to do with threading
        self.stop = False
        self.frames = []  # initiatlize frames
        atexit.register(self.close)

    def new_frame(self, data, frame_count, time_info, status):
        """
        function to call per "frame" generated
        each frame has "data"
        """
        data = np.fromstring(data, 'int16')
        with self.lock:  # using threading?
            self.frames.append(data)  # add data to the array of "frames"
            if self.stop:
                return None, pyaudio.paComplete
        return None, pyaudio.paContinue

    def get_frames(self):
        with self.lock:  # using threading?
            frames = self.frames  # return the frames accumulated - should have only one
            self.frames = []  # clear frames
            return frames

    def start(self):
        self.stream.start_stream()  # opening recording stream

    def close(self):  # some closing procedure, perhaps to erase memory
        with self.lock:
            self.stop = True
        self.stream.close()
        self.p.terminate()


class MplFigure(object):  # don't know what is this for
    def __init__(self, parent):
        self.figure = plt.figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, parent)

class LiveFFTWidget(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)

        self.chunksize = 2048
        self.tempo_res = 32  # r_coeff resolution, needs to be a factor of chunksize
        self.iteration = 0  # for counting, if needed
        self.noise = np.round(200000*np.random.randn(self.chunksize))  # to desensitise onset detection
        self.sampling_rate = 44100
        self.notes_dict = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
                           6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}

        # holding variables
        self.signal_frame_pp0 = [0] * self.chunksize          # past signal chunk
        self.signal_frame_pp1 = [0] * self.chunksize
        self.signal_frame_pp2 = [0] * self.chunksize     # past past signal chunk
        self.energy_frame_pp0 = [0] * self.chunksize          # past energy chunk
        self.energy_frame_pp1 = [0] * self.chunksize  # past energy chunk
        self.energy_frame_pp2 = [0] * self.chunksize  # past energy chunk
        self.rcoeff_frame_pp0 = [0.0] * int(self.tempo_res)   # current rcoeff chunk - should change
        self.rcoeff_frame_pp1 = [0.0] * int(self.tempo_res)  # past roceff chunk
        self.note_detected = False

        # customize the UI
        self.initUI()

        # init class data
        self.initData()

        # connect slots
        self.connectSlots()  # don't know what is this for

        # init MPL widget
        self.initMplWidget()  # (refer to MplFigure class)

        self.start_time = time.time()  # start timer
        self.prev_time = time.time()  # to calculate the time difference

    def initUI(self):  # comment on this later
        hbox_gain = QtGui.QHBoxLayout()
        autoGain = QtGui.QLabel('Auto gain')
        autoGainCheckBox = QtGui.QCheckBox(checked=True)
        hbox_gain.addWidget(autoGain)
        hbox_gain.addWidget(autoGainCheckBox)

        # reference to checkbox
        self.autoGainCheckBox = autoGainCheckBox

        hbox_fixedGain = QtGui.QHBoxLayout()
        fixedGain = QtGui.QLabel('Fixed gain level')
        fixedGainSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        hbox_fixedGain.addWidget(fixedGain)
        hbox_fixedGain.addWidget(fixedGainSlider)

        self.fixedGainSlider = fixedGainSlider

        vbox = QtGui.QVBoxLayout()

        vbox.addLayout(hbox_gain)
        vbox.addLayout(hbox_fixedGain)

        # mpl figure
        self.main_figure = MplFigure(self)
        vbox.addWidget(self.main_figure.toolbar)
        vbox.addWidget(self.main_figure.canvas)

        self.setLayout(vbox)

        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('LiveFFT')
        self.show()

        # timer for calls, taken from:
        # http://ralsina.me/weblog/posts/BB974.html
        timer = QtCore.QTimer()
        timer.timeout.connect(self.handleNewData)  # calls handleNewData every 20ms
        timer.start(20)  # chunks come out at a frequency of approximately 46ms
        # keep reference to timer
        self.timer = timer

    def initData(self):
        mic = MicrophoneRecorder(rate=44100, chunksize=self.chunksize)
        mic.start()

        # keeps reference to mic
        self.mic = mic

        # computes the parameters that will be used during plotting
        self.freq_vect = np.fft.rfftfreq(mic.chunksize, 1./mic.rate)  # original
        self.time_vect = np.arange(mic.chunksize, dtype=np.float32) / mic.rate * 1000
        # QUESTION: Why 1000? - convert from seconds to milliseconds?
        # these are axes that we will plot against

    def connectSlots(self):
        pass  # don't know what is this for

    def initMplWidget(self):
        """
        creates initial matplotlib plots in the main window and keeps
        references for further use
        """
        # top plot: currently to show energy
        self.ax_top = self.main_figure.figure.add_subplot(211)
        self.ax_top.set_ylim(-32768, 32768)  # original
        self.ax_top.set_ylim(-32768 * 100, 32768 * 100)  # to show energy
        self.ax_top.set_xlim(0, self.time_vect.max())
        self.ax_top.set_xlabel(u'time (ms)', fontsize=6)

        # bottom plot: currently to show spectrum
        self.ax_bottom = self.main_figure.figure.add_subplot(212)
        self.ax_bottom.set_ylim(0, 1)
        # self.ax_bottom.set_xlim(0, self.freq_vect.max()) original
        self.ax_bottom.set_xlim(0, 5000.)
        self.ax_bottom.set_xlabel(u'frequency (Hz)', fontsize=6)

        # line objects
        self.line_top, = self.ax_top.plot(self.time_vect,
                                          np.ones_like(self.time_vect), lw=0.5)

        self.line_bottom, = self.ax_bottom.plot(self.freq_vect,
                                                np.ones_like(self.freq_vect), lw=0.5)

        self.pitch_line, = self.ax_bottom.plot((self.freq_vect[self.freq_vect.size / 2],
                                                self.freq_vect[self.freq_vect.size / 2]),
                                               self.ax_bottom.get_ylim(), lw=2)
        # This plots for vertical line that marks the pitch

        # tight layout
        #plt.tight_layout()

    def handleNewData(self):
        """ handles the asynchronously collected sound chunks """

        # print str(time.time() - self.start_time) + "  " + str(time.time() - self.prev_time) + \
        # " gets the latest frames"
        # self.prev_time = time.time()
        signal_frames = self.mic.get_frames()

        # print str(time.time() - self.start_time) + "  " + str(time.time() - self.prev_time) + \
        # " taking last frame"
        # self.prev_time = time.time()
        if len(signal_frames) > 0:
            if len(signal_frames) > 1:
                print str(len(signal_frames) - 1) + " frame lost"
                # indicate number of frames lost - should not have any
            self.signal_frame_pp0 = signal_frames[-1]  # keeps only the last frame

            # print str(time.time() - self.start_time) + "  " + str(time.time() - self.prev_time) + \
            # " energy calculations"  # 0.01s
            # self.prev_time = time.time()
            # to calculate the rectangular window for every sample
            # numpy operations are more efficient than using python loops
            # the size of the rectangular window is one chunksize
            # convolution can be considered
            self.energy_frame_pp0 = np.full(self.chunksize, sum(np.absolute(self.signal_frame_pp0)), dtype="int32")
            to_cumsum = np.add(np.absolute(self.signal_frame_pp0), -np.absolute(self.signal_frame_pp1))
            cumsum = np.cumsum(to_cumsum)
            self.energy_frame_pp0[1:] = np.add(self.energy_frame_pp0[1:], cumsum[:-1])
            self.energy_frame_pp0 = np.add(self.energy_frame_pp0,self.noise)

            # print str(time.time() - self.start_time) + "  " + str(time.time() - self.prev_time) + \
            # " r_coeff calculations"
            # self.prev_time = time.time()
            # calculating pearson correlation coefficient at 2048/32 samples
            # to determine exact time of onset
            # could not think of any way this could be done faster
            for i in range(self.tempo_res):
                energy_arg = np.concatenate((self.energy_frame_pp1[i*self.chunksize/self.tempo_res:],
                                             self.energy_frame_pp0[:-(self.tempo_res-i)*self.chunksize/self.tempo_res]))
                self.rcoeff_frame_pp0[i] = np.corrcoef(energy_arg, np.arange(self.chunksize))[0,1]

            # print str(time.time() - self.start_time) + "  " + str(time.time() - self.prev_time) + \
            # " detecting new note"
            # self.prev_time = time.time()
            #print self.rcoeff_frame_pp0
            rcoeff_arg = np.concatenate((self.rcoeff_frame_pp1, self.rcoeff_frame_pp0))
            # we need the previous rcoeff frame to determine onset

            # finding the onset
            for i in range(self.tempo_res, 0, -1):
                if rcoeff_arg[-i] > 0.80 and all(i < 0.80 for i in rcoeff_arg[-i-31:-i]):
                    # to determine crossing point,
                    # 30 entries cooldown - check that previous entries do not have cooldown
                    # TODO: check whether are we looking at the correct time
                    #print "NEW NOTE at " + str(time.time() - self.start_time)

                    # print str(time.time() - self.start_time) + "  " + str(time.time() - self.prev_time) + \
                    # " note class"
                    # self.prev_time = time.time()
                    time_arg = np.concatenate((self.signal_frame_pp2, self.signal_frame_pp1, self.signal_frame_pp0))
                    time_arg = time_arg[-i*self.chunksize/self.tempo_res-self.chunksize:-i*self.chunksize/self.tempo_res]

                    # retired code using cqt from librosa
                    # float = 1.0 / 32768.0 * np.array(time_arg)
                    # y_harm, y_perc = librosa.effects.hpss(float)
                    # chroma = librosa.feature.chroma_cqt(y=y_harm, sr=self.sampling_rate, hop_length=65536)
                    # max = np.argmax(chroma)
                    # note = self.notes_dict[max]

                    # following is the hps algorithm
                    # TODO: take the difference between the current chunk and the previous chunk
                    spectrum = np.absolute(np.fft.fft(time_arg))
                    spectrum[:10] = 0.0  # anything below middle C is muted
                    spectrum[1024:] = 0.0  # mute second half of spectrum, lazy to change code

                    scale1 = [0.0] * (2048 * 6)
                    scale2 = [0.0] * (2048 * 6)
                    scale3 = [0.0] * (2048 * 6)

                    # upsampling the original scale spectrum, 6 for 1
                    scale1_f1 = np.convolve(spectrum, [5.0 / 6.0, 1.0 / 6.0])[1:]
                    scale1_f2 = np.convolve(spectrum, [4.0 / 6.0, 2.0 / 6.0])[1:]
                    scale1_f3 = np.convolve(spectrum, [3.0 / 6.0, 3.0 / 6.0])[1:]
                    scale1_f4 = np.convolve(spectrum, [2.0 / 6.0, 4.0 / 6.0])[1:]
                    scale1_f5 = np.convolve(spectrum, [1.0 / 6.0, 5.0 / 6.0])[1:]
                    scale1[::6] = spectrum
                    scale1[1::6] = scale1_f5
                    scale1[2::6] = scale1_f4
                    scale1[3::6] = scale1_f3
                    scale1[4::6] = scale1_f2
                    scale1[5::6] = scale1_f1
                    # downsampling from the 6 for 1 upsample
                    scale2[:2048 * 3] = scale1[::2]
                    scale3[:2048 * 2] = scale1[::3]
                    hps = np.prod((scale1, scale2, scale3), axis=0)  # the "product" in harmonic product spectrum
                    hps_max = np.argmax(hps)  # determine the location of the peak of hps result
                    ffreq = hps_max * 44100.0 / (2048.0 * 6.0)  # calculate the corresponding frequency of the peak
                    # TODO: carry out some checks that this note is indeed feasible
                    # forumla: sampling rate / (chunksize * upsampling value)
                    note_no = (np.log2(ffreq) - np.log2(220.0)) * 12.0  # take logarithm and find note
                    note_no_rounded = np.round(note_no)  # round off to nearest note
                    # TODO: use notes dict to print the note in solfage form

                    print str(ffreq) + ", " + str(note_no) + ", " + str(note_no_rounded) + \
                          " at " + str(time.time() - self.start_time)
                    self.note_detected = True

            # print str(time.time() - self.start_time) + "  " + str(time.time() - self.prev_time) + \
            # " storing for recursion"
            # self.prev_time = time.time()
            self.energy_frame_pp1 = self.energy_frame_pp0
            self.signal_frame_pp2 = self.signal_frame_pp1
            self.signal_frame_pp1 = self.signal_frame_pp0
            self.rcoeff_frame_pp1 = self.rcoeff_frame_pp0

            display_only_note = False
            if self.note_detected or not display_only_note:
                # print str(time.time() - self.start_time) + "  " + str(time.time() - self.prev_time) + \
                # " set time on graph"  # 0.001s
                # self.prev_time = time.time()
                # plots the time signal
                self.line_top.set_data(self.time_vect, self.energy_frame_pp0)

                # print str(time.time() - self.start_time) + "  " + str(time.time() - self.prev_time) + \
                # " take FFT"
                # self.prev_time = time.time()
                # computes and plots the fft signal
                fft_frame = np.fft.rfft(self.signal_frame_pp0)

                # print str(time.time() - self.start_time) + "  " + str(time.time() - self.prev_time) + \
                # " some thing about scaling"
                # self.prev_time = time.time()
                # inherited, don't know what is this for
                # perhaps it is to normalise the spectrum - plotting is faster without changing axes
                if self.autoGainCheckBox.checkState() == QtCore.Qt.Checked:
                    fft_frame /= np.abs(fft_frame).max()
                else:
                    fft_frame *= (1 + self.fixedGainSlider.value()) / 5000000.
                    # print(np.abs(fft_frame).max())

                # print str(time.time() - self.start_time) + "  " + str(time.time() - self.prev_time) + \
                # " set spectrum on graph"  # 0.001 s
                # self.prev_time = time.time()
                self.line_bottom.set_data(self.freq_vect, np.abs(fft_frame))

                # print str(time.time() - self.start_time) + "  " + str(time.time() - self.prev_time) + \
                # " placeholder"  # 0.8s wdf
                # self.prev_time = time.time()

                new_pitch = 8.0  # to make this information meaningful
                precise_pitch = 10.0

                # print str(time.time() - self.start_time) + "  " + str(time.time() - self.prev_time) + \
                # " set pitch on graph"
                # self.prev_time = time.time()
                self.ax_bottom.set_title("pitch = {:.2f} Hz".format(precise_pitch))
                self.pitch_line.set_data((new_pitch, new_pitch),
                                         self.ax_bottom.get_ylim())  # move the vertical pitch line

                if self.iteration % 1 == 0:  # update plot only after every n chunks, if necessary
                    # print str(time.time() - self.start_time) + "  " + str(time.time() - self.prev_time) + \
                    # " refresh plot"  # 0.105s
                    # self.prev_time = time.time()
                    self.main_figure.canvas.draw()  # refreshes the plots, takes the bulk of time

                self.note_detected = False

        self.iteration += 1
        #print str(time.time() - self.start_time) + "  " + str(time.time() - self.prev_time) + \
        # " end of loop \n \n"
        # self.prev_time = time.time()


def downsample(pwm, fraction, signal_length):
    """downsample n to 1"""
    convolve_result = [0.0] * int(signal_length / 2)
    # convolve_result = [1.0] * int(signal_length / 2)
    convolve_full = np.convolve(pwm, [1.0 / fraction] * fraction)[fraction - 1::fraction]
    convolve_result[:len(convolve_full)] = convolve_full
    return convolve_result

def compute_pitch_hps(x, Fs, dF=None, Fmin=30., Fmax=900., H=5):
    """
    original hps alogrithm, which has a long running time
    don't know how the last part of this algorithm works
    """
    # default value for dF frequency resolution
    if dF == None:
        dF = Fs / x.size

    # Hamming window apodization
    x = np.array(x, dtype=np.double, copy=True)
    x *= np.hamming(x.size)

    # number of points in FFT to reach the resolution wanted by the user
    n_fft = np.ceil(Fs / dF)

    # DFT computation
    X = np.abs(np.fft.fft(x, n=int(n_fft)))

    # limiting frequency R_max computation
    R = np.floor(1 + n_fft / 2. / H)

    # computing the indices for min and max frequency
    N_min = np.ceil(Fmin / Fs * n_fft)
    N_max = np.floor(Fmax / Fs * n_fft)
    N_max = min(N_max, R)

    # harmonic product spectrum computation - wdf is going on here?
    indices = (np.arange(N_max)[:, np.newaxis] * np.arange(1, H+1)).astype(int)
    P = np.prod(X[indices.ravel()].reshape(N_max, H), axis=1)
    ix = np.argmax(P * ((np.arange(P.size) >= N_min) & (np.arange(P.size) <= N_max)))
    return dF * ix

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = LiveFFTWidget()
    sys.exit(app.exec_())
