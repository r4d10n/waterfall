#!/usr/bin/env python

# Copyright (c) 2019 Jeppe Ledet-Pedersen
# This software is released under the MIT license.
# See the LICENSE file for further details.

import sys
import json
import argparse
import os
import json
import time

from gnuradio import gr
from gnuradio import iio
from gnuradio import uhd
from gnuradio.fft import logpwrfft

import numpy as np

from gevent.pywsgi import WSGIServer
from geventwebsocket import WebSocketError
from geventwebsocket.handler import WebSocketHandler

from bottle import request, Bottle, abort, static_file

app = Bottle()
connections = set()
opts = {}

@app.route('/websocket')
def handle_websocket():
    global wsock
    wsock = request.environ.get('wsgi.websocket')
    if not wsock:
        abort(400, 'Expected WebSocket request.')

    connections.add(wsock)

    # Send center frequency and span
    wsock.send(json.dumps(opts))

    while True:
        try:
          message = json.loads(wsock.receive())
          for key, value in message.items():
            
            if key == "freq":
                set_frequency(value)
            elif key == "span":
                set_span(value)
            elif key == "gain":
                set_gain(value)
            elif key == "fps":
                set_fps(value)

          #print("Message:" + str(message["freq"]))
        except TypeError:
            break
        except WebSocketError:
            break

    connections.remove(wsock)


@app.route('/')
def index():
    return static_file('index.html', root='.')


@app.route('/<filename>')
def static(filename):
    return static_file(filename, root='.')

def set_frequency(freq):            #handle UHD
    if freq > 70e6 and freq < 6e9:
        opts['center'] = freq
        print ("freq: " + str(freq))
        update_opts()

def set_span(span):
    if span > 3e6 and span < 61e6:
        opts['span'] = span
        update_opts()

def set_gain(gain):
    if gain > 0 and gain < 72:
        opts['gain'] = gain
        update_opts()

def set_fps(fps):
    if fps > 0 and fps < 2000:
        opts['framerate'] = fps
        update_opts()
        
def update_opts():                  # handle UHD
    tb.set_source_params(opts)
    tb.set_fft_params(opts)
    wsock.send(json.dumps(opts))    
    
class fft_broadcast_sink(gr.sync_block):
    def __init__(self, fft_size):
        gr.sync_block.__init__(self,
                               name="plotter",
                               in_sig=[(np.float32, fft_size)],
                               out_sig=[])

    def work(self, input_items, output_items):
        ninput_items = len(input_items[0])

        for bins in input_items[0]:
            p = np.around(bins).astype(int)
            p = np.fft.fftshift(p)
            for c in connections.copy():
                try:
                    c.send(json.dumps({'s': p.tolist()}, separators=(',', ':')))
                except Exception:
                    connections.remove(c)

        self.consume(0, ninput_items)

        return 0


class fft_receiver(gr.top_block):
    def __init__(self, uri, samp_rate, freq, bw, gain, fft_size, framerate):
        gr.top_block.__init__(self, "Top Block")

        self.fft = logpwrfft.logpwrfft_c(
            sample_rate=samp_rate,
            fft_size=fft_size,
            ref_scale=1,
            frame_rate=framerate,
            avg_alpha=1,
            average=False,
        )
        self.fft_broadcast = fft_broadcast_sink(fft_size)

        self.connect((self.fft, 0), (self.fft_broadcast, 0))

        try:
            self.pluto_source = iio.pluto_source(uri, int(freq), int(samp_rate), int(bw), 0x8000, True, True, True, "manual", gain, '', True)
            self.connect((self.pluto_source, 0), (self.fft, 0))
            return None
        except RuntimeError as e:
            print("IIO Device not found....")
        
        try:
            self.usrp = uhd.usrp_source(
                    ",".join(("", "")),
                    uhd.stream_args(
                        cpu_format="fc32",
                        channels=range(1),
                        ),
                    )
            self.usrp.set_samp_rate(samp_rate)
            self.usrp.set_center_freq(freq, 0)
            self.usrp.set_gain(gain, 0)
            self.connect((self.usrp, 0), (self.fft, 0))
            return None
        except RuntimeError as e:
            print("UHD Device not found...\n")            
            exit()
    
    def set_source_params(self, opts):
        tic = time.time()                           # optimize tune time with direct IIO tune functions
        self.pluto_source.set_params(int(opts['center']), int(opts['span']), int(opts['bw']), True, True, True, "manual", int(opts['gain']), '', True)
        toc = time.time()
        print("Setting Pluto parameters in " + str(toc-tic) + " secs...\n")

    def set_fft_params(self, opts):
        self.fft.set_sample_rate(opts['span'])
        self.fft.set_vec_rate(opts['framerate'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--uri', default='ip:192.168.2.1')
    parser.add_argument('-s', '--sample-rate', type=float, default=60e6)
    parser.add_argument('-f', '--frequency', type=float, default=940e6)
    parser.add_argument('-g', '--gain', type=float, default=71)
    parser.add_argument('-b', '--bw', type=float, default=56e6)
    parser.add_argument('-n', '--fft-size', type=int, default=4096)
    parser.add_argument('-r', '--frame-rate', type=int, default=25)

    args = parser.parse_args()

    if gr.enable_realtime_scheduling() != gr.RT_OK or 0:
        print("Error: failed to enable real-time scheduling.")

    global tb

    tb = fft_receiver(
        uri=args.uri,
        samp_rate=args.sample_rate,
        freq=args.frequency,
        gain=args.gain,
        bw=args.bw,
        fft_size=args.fft_size,
        framerate=args.frame_rate
    )
    tb.start()

    opts['center'] = args.frequency
    opts['span'] = args.sample_rate
    opts['uri'] = args.uri
    opts['gain']= args.gain
    opts['bw'] = args.bw
    opts['fft_size'] = args.fft_size
    opts['framerate'] = args.frame_rate

    server = WSGIServer(("0.0.0.0", 8000), app,
                        handler_class=WebSocketHandler)
    try:
        server.serve_forever()
    except Exception:
        sys.exit(0)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
