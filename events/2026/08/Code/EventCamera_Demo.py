import sensor
import time
from pyb import UART, LED

uart = UART(3, 115200)
led = LED(1)

sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.B320X320)

clock = time.clock()

event_counter = 0

while True:
    clock.tick()

    img = sensor.snapshot()

    # 仮のイベント量っぽい値
    event_counter = (event_counter + 3) & 0xFF

    uart.write(bytes([event_counter]))

    led.toggle()

    print(clock.fps())
