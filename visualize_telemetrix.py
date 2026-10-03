#!/usr/bin/env python3
"""Live Arduino Uno input monitor; run with --help for options."""
import argparse
from collections import deque
import math
from queue import Empty, Queue
import threading
import time
import tkinter as tk
from tkinter import ttk


SERVO_PIN = 9
DIGITAL_PINS = tuple(pin for pin in range(2, 14) if pin != SERVO_PIN)


class Monitor:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.board = None
        self.servo_ready = threading.Event()
        self.servo_commands = Queue()
        self.servo_status = "Nie wysłano pozycji"
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.values = {}
        self.status = 'Łączenie z Arduino…'
        self.history = {pin: deque(maxlen=300) for pin in range(6)}
        self.digital = {}
        self.analog = {}
        root.title('Arduino Uno — Telemetrix: wejścia')
        root.geometry('960x760')
        root.protocol('WM_DELETE_WINDOW', self.close)
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='Arduino Uno · monitor wejść', font=('', 20, 'bold')).pack(anchor='w')
        self.status_label = ttk.Label(frame, text=self.status)
        self.status_label.pack(anchor='w', pady=(4, 12))
        digital_frame = ttk.LabelFrame(frame, text='Cyfrowe D2–D13 (bez D9) · ' + ('INPUT_PULLUP' if args.pullup else 'INPUT'), padding=8)
        digital_frame.pack(fill='x')
        for pin in DIGITAL_PINS:
            label = tk.Label(digital_frame, text=f'D{pin}\n—', bg='#e2e8f0', width=6, pady=10)
            label.grid(row=0, column=pin-2, padx=3, sticky='ew')
            digital_frame.columnconfigure(pin-2, weight=1)
            self.digital[pin] = label
        ttk.Label(frame, text='Szary: brak raportu · zielony: HIGH (1) · ciemny: LOW (0)').pack(anchor='w', pady=8)
        servo_frame = ttk.LabelFrame(frame, text='Serwo · D9', padding=8)
        servo_frame.pack(fill='x', pady=(0, 8))
        self.servo_buttons = []
        for angle in (0, 90, 180):
            button = ttk.Button(servo_frame, text=f'Ustaw {angle}°', state='disabled',
                                command=lambda angle=angle: self.set_servo(angle))
            button.pack(side='left', padx=4)
            self.servo_buttons.append(button)
        self.servo_label = ttk.Label(servo_frame, text=self.servo_status)
        self.servo_label.pack(side='left', padx=12)
        analog_frame = ttk.LabelFrame(frame, text='Analogowe A0–A5 · ADC 0–1023 · ostatnie 30 s', padding=8)
        analog_frame.pack(fill='both', expand=True)
        analog_frame.columnconfigure(2, weight=1)
        for pin in range(6):
            analog_frame.rowconfigure(pin, weight=1)
            label = ttk.Label(analog_frame, text=f'A{pin}: —', width=25)
            label.grid(row=pin, column=0, padx=5)
            bar = ttk.Progressbar(analog_frame, maximum=1023, length=130)
            bar.grid(row=pin, column=1, padx=8)
            canvas = tk.Canvas(analog_frame, height=65, bg='#0f172a', highlightthickness=0)
            canvas.grid(row=pin, column=2, sticky='nsew', padx=5, pady=5)
            self.analog[pin] = (label, bar, canvas)
        ttk.Label(frame, text=f'Napięcie szacowane dla Vref = {args.vref:g} V. Odświeżanie: 10 Hz.').pack(anchor='w', pady=(8, 0))
        self.worker = threading.Thread(target=self.connect, daemon=True)
        self.worker.start()
        self.refresh()

    def callback(self, kind):
        def receive(data):
            with self.lock:
                self.values[kind, data[1]] = data[2]
        return receive

    def set_servo(self, angle):
        if self.servo_ready.is_set() and not self.stop.is_set():
            self.servo_commands.put(angle)

    def process_servo(self):
        try:
            angle = self.servo_commands.get_nowait()
        except Empty:
            return
        if not self.args.demo:
            self.board.servo_write(SERVO_PIN, angle)
        prefix = 'DEMO · zadano' if self.args.demo else 'Wysłano pozycję'
        self.servo_status = f'{prefix}: {angle}°'

    def connect(self):
        try:
            if self.args.demo:
                self.status = 'DEMO — dane symulowane, bez połączenia z Arduino'
                self.servo_ready.set()
                while not self.stop.wait(0.05):
                    self.process_servo()
                    now = time.monotonic()
                    with self.lock:
                        for pin in DIGITAL_PINS:
                            self.values['digital', pin] = int(now / (0.3 + pin * .1)) % 2
                        for pin in range(6):
                            self.values['analog', pin] = round(511.5 * (1 + math.sin(now + pin)))
                return
            from telemetrix import telemetrix
            self.board = telemetrix.Telemetrix(com_port=self.args.port)
            digital_mode = (self.board.set_pin_mode_digital_input_pullup if self.args.pullup
                            else self.board.set_pin_mode_digital_input)
            for pin in DIGITAL_PINS:
                if self.stop.is_set():
                    return
                digital_mode(pin, callback=self.callback('digital'))
                time.sleep(.02)
            for pin in range(6):
                if self.stop.is_set():
                    return
                self.board.set_pin_mode_analog_input(pin, differential=1, callback=self.callback('analog'))
                time.sleep(.02)
            if self.stop.is_set():
                return
            self.board.set_pin_mode_servo(SERVO_PIN)
            self.servo_ready.set()
            self.status = 'Skonfigurowano wejścia i serwo D9 · oczekiwanie na raporty zmian'
            while not self.stop.wait(0.05):
                self.process_servo()
        except Exception as exc:
            self.status = f'Błąd: {exc}'
        finally:
            self.servo_ready.clear()
            if self.board is not None:
                try:
                    self.board.shutdown()
                except Exception as exc:
                    self.status = f'Błąd zamykania: {exc}'

    def refresh(self):
        if self.stop.is_set():
            return
        with self.lock:
            values = dict(self.values)
        self.status_label.config(text=self.status)
        self.servo_label.config(text=self.servo_status)
        for button in self.servo_buttons:
            button.config(state='normal' if self.servo_ready.is_set() else 'disabled')
        for pin, label in self.digital.items():
            value = values.get(('digital', pin))
            if value is not None:
                label.config(text=f'D{pin}\n{value} / {"HIGH" if value else "LOW"}',
                             bg='#22c55e' if value else '#334155', fg='white')
        now = time.monotonic()
        for pin, (label, bar, canvas) in self.analog.items():
            value = values.get(('analog', pin))
            history = self.history[pin]
            history.append((now, value))
            if value is not None:
                label.config(text=f'A{pin}: {value:4d}  ≈ {value * self.args.vref / 1023:.3f} V')
                bar['value'] = value
            width, height = canvas.winfo_width(), canvas.winfo_height()
            canvas.delete('all')
            canvas.create_line(0, height / 2, width, height / 2, fill='#334155')
            points = []
            for stamp, sample in history:
                if sample is not None and now - stamp <= 30:
                    points.extend((width * (1 - (now - stamp) / 30),
                                   height - 3 - sample / 1023 * (height - 6)))
            if len(points) >= 4:
                canvas.create_line(*points, fill='#38bdf8', width=2)
        self.root.after(100, self.refresh)

    def close(self):
        self.stop.set()
        for button in self.servo_buttons:
            button.config(state='disabled')
        self.status_label.config(text='Zamykanie połączenia…')
        self.wait_closed()

    def wait_closed(self):
        if self.worker.is_alive():
            self.root.after(100, self.wait_closed)
        else:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', help='Port szeregowy, np. /dev/ttyACM0 lub COM3; domyślnie autodetekcja')
    parser.add_argument('--pullup', action='store_true', help='Włącz rezystory podciągające wejść cyfrowych')
    parser.add_argument('--vref', type=float, default=5.0, help='Napięcie odniesienia ADC w V (domyślnie 5)')
    parser.add_argument('--demo', action='store_true', help='Symulacja bez Arduino i biblioteki Telemetrix')
    args = parser.parse_args()
    if not math.isfinite(args.vref) or args.vref <= 0:
        parser.error('--vref musi być dodatnią, skończoną liczbą')
    root = tk.Tk()
    app = Monitor(root, args)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.close()
        root.mainloop()


if __name__ == '__main__':
    main()
