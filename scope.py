import serial
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# --- Settings ---
SERIAL_PORT = "COM3"
BAUDRATE = 115200
ADC_CHANNEL_COUNT = 2
SYNC_WORD = b'\xA5\xA5'
BYTES_PER_CHANNEL = 2
DATA_PAYLOAD_SIZE = ADC_CHANNEL_COUNT * BYTES_PER_CHANNEL
EXPECTED_PACKET_SIZE = len(SYNC_WORD) + DATA_PAYLOAD_SIZE # = 6 bytes

VREF = 3.3
ADC_RESOLUTION = 12
ADC_MAX_VALUE = 2**ADC_RESOLUTION

# Plotting Settings
PLOT_WINDOW_SIZE = 100 # Number of points to display on the plot
# UPDATE_INTERVAL_MS: Determines how often the plot ATTEMPTS TO REDRAW.
# Instead of setting it too low (e.g., 1ms), keeping it at a reasonable value (15-30ms) usually works better.
# Because the main speed-up comes from skipping data.
UPDATE_INTERVAL_MS = 20
# Determine the maximum number of packets to read in a single update period
# This prevents the update function from getting stuck for too long.
MAX_READS_PER_FRAME = 25 # You can adjust this value according to your data rate

# --- Data Storage ---
time_data = deque(maxlen=PLOT_WINDOW_SIZE)
voltage_data_ch1 = deque(maxlen=PLOT_WINDOW_SIZE)
voltage_data_ch2 = deque(maxlen=PLOT_WINDOW_SIZE)
# This counter will now only keep track of the number of plotted points
plotted_sample_index = 0

# --- Serial Port Connection ---
ser = None
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.01) # We can set the timeout very low
    print(f"Connected to {SERIAL_PORT} at {BAUDRATE}bps.")
    ser.reset_input_buffer()
    time.sleep(0.5)
except serial.SerialException as e:
    print(f"Failed to open serial port {SERIAL_PORT}: {e}")
    exit()

# --- Plot Setup ---
fig, ax = plt.subplots()
# Adding animated=True is important for blitting
line1, = ax.plot([], [], lw=2, label='Channel 1 (A0/CH0)', animated=True)
line2, = ax.plot([], [], lw=2, label='Channel 2 (A1/CH1)', animated=True)
ax.set_title('Real-Time ADC Readings (Fast Update)')
ax.set_xlabel('Plotted Sample Number') # Updated the axis label
ax.set_ylabel('Voltage (V)')
ax.legend(loc='upper left')
ax.grid(True)
ax.set_ylim(-0.1, VREF + 0.2)
ax.set_xlim(0, PLOT_WINDOW_SIZE)

unpack_format = '<' + 'H' * ADC_CHANNEL_COUNT

# --- Synchronization Function ---
# This function is still necessary, but should only be called at the start or in case of an error.
# We could potentially merge it with the logic inside the update function.
def find_sync(ser_instance):
    print("Attempting to find sync word...")
    sync_buf = bytearray(len(SYNC_WORD))
    try:
        ser_instance.reset_input_buffer()
        timeout_start = time.time()
        while time.time() - timeout_start < 1.0: # Search for sync for a maximum of 1 second
            if ser_instance.in_waiting > 0:
                byte = ser_instance.read(1)
                if not byte: continue
                sync_buf = sync_buf[1:] + byte
                if sync_buf == SYNC_WORD:
                    print("Sync word found!")
                    return True
            else:
                time.sleep(0.005) # Short sleep to avoid high CPU usage
        print("Sync word not found within timeout.")
        return False
    except Exception as e:
        print(f"Error during sync search: {e}")
        return False

# --- Animation Update Function (Optimized) ---
def update(frame):
    global plotted_sample_index, ser

    if ser is None or not ser.is_open:
        return [] # Return an empty list if the port is closed (important for blit)

    last_voltage_ch1 = None
    last_voltage_ch2 = None
    packet_found_in_frame = False

    try:
        # Read as long as there is data in the buffer and the max read limit hasn't been reached
        for _ in range(MAX_READS_PER_FRAME):
            # Check if there is enough data in the buffer (Sync+Data)
            if ser.in_waiting >= EXPECTED_PACKET_SIZE:
                # First read (or check) the Sync Word
                sync_bytes = ser.read(len(SYNC_WORD))
                if len(sync_bytes) < len(SYNC_WORD): break # Timeout etc.

                if sync_bytes == SYNC_WORD:
                    # Sync is correct, now read the data payload
                    data_payload = ser.read(DATA_PAYLOAD_SIZE)
                    if len(data_payload) == DATA_PAYLOAD_SIZE:
                        # Unpack the data
                        raw_values = struct.unpack(unpack_format, data_payload)
                        # Only store the last read values
                        last_voltage_ch1 = (raw_values[0] / float(ADC_MAX_VALUE)) * VREF
                        last_voltage_ch2 = (raw_values[1] / float(ADC_MAX_VALUE)) * VREF
                        packet_found_in_frame = True # At least 1 packet processed in this frame
                    else:
                        # Incomplete data after sync? Synchronization might be lost.
                        print("Incomplete data after sync.")
                        # It might be good to clear the buffer and start over
                        ser.reset_input_buffer()
                        break # End this frame

                else:
                    # Expected Sync Word not received. Clear buffer and search for sync.
                    print(f"Sync lost. Expected A5A5, got {sync_bytes.hex()}. Re-syncing...")
                    # We can call find_sync or simply clear the buffer and let the next frame handle it.
                    ser.reset_input_buffer()
                    break # End this frame
            else:
                # Not enough data in the buffer, exit the read loop
                break

        # After the read loop, if data was found in this frame, update the plot
        if packet_found_in_frame:
            time_data.append(plotted_sample_index)
            voltage_data_ch1.append(last_voltage_ch1)
            voltage_data_ch2.append(last_voltage_ch2)
            plotted_sample_index += 1

            line1.set_data(list(time_data), list(voltage_data_ch1))
            line2.set_data(list(time_data), list(voltage_data_ch2))

            if plotted_sample_index > PLOT_WINDOW_SIZE:
                ax.set_xlim(plotted_sample_index - PLOT_WINDOW_SIZE, plotted_sample_index)
            else:
                ax.set_xlim(0, PLOT_WINDOW_SIZE)

            return line1, line2 # Return only the updated lines

    except (serial.SerialException, OSError) as e:
        print(f"Serial error in update loop: {e}. Closing port.")
        if ser and ser.is_open: ser.close()
        ser = None
    except Exception as e:
        print(f"Unexpected error in update: {e}")
        # It might be good to clear the buffer in case of an error too
        if ser and ser.is_open: ser.reset_input_buffer()

    # Return an empty list if no data was processed in this frame or if there was an error
    return []


# --- Start Animation ---
# blit=True is important!
ani = animation.FuncAnimation(fig, update, interval=UPDATE_INTERVAL_MS, blit=True, save_count=50)

# --- Show Plot ---
try:
    plt.show()
except Exception as e:
    print(f"Error displaying plot: {e}")

# --- Cleanup ---
if ser is not None and ser.is_open:
    ser.close()
    print("Serial port closed.")

print("Program finished.")
