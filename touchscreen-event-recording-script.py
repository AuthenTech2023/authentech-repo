# Import Libraries
import subprocess, os, time, csv

# Display Launch Prompt
print("Welcome to ATA (Android Touch Acquisition)")
print()
os.chdir("C:/Users/lr7247es/Desktop/Recording script/adb") #directory of adb
# Retrieve List of Devices
devices = subprocess.run(['./adb.exe', 'devices'], stdout=subprocess.PIPE)
devices = devices.stdout.decode('utf-8').strip()
devices = devices.replace("\tdevice", "")
devices = devices.split("\r\n")[1:]
# Prompt User to Select Device
print("Select a Device: ")
for i in range(len(devices)):
    print(str(i + 1) + ") " + devices[i])

device = int(input("Enter Choice: "))
device = devices[device - 1]
print("Selected Device: " + device)
# Prompt User to Specify Session Length
session_time = int(input("Specify Session Length (min): "))
session_time = session_time * 60  # Convert to minutes
# Prompt User for Application to launch

# Prompt User For Event Device
# event_id = input("Enter Device Event ID (LG=1): ")  
event_id = str(5)    
# Start Tracking Session
os.chdir("M:/My_Private_Files/Event Recordings") #Directory to save recordings
user_id = input(
    "Enter ID for User *Warning, make sure it doesn't conflict with other filenames*:")  
log_file = open(user_id, "w")
os.chdir("C:/Users/lr7247es/Desktop/Recording script/adb")
data = subprocess.Popen(['./adb.exe', '-s', device, 'shell', 'getevent -lt /dev/input/event' + event_id],
                        stdout=log_file)
# Pipe log output to file. Show time remaining for session, prompt to end session early.
print("***Tracking Session Started***")
while (session_time > 0):
    print("Time Remaining: " + str(session_time) + " seconds.", end="\r", flush=True)
    time.sleep(1)
    session_time -= 1
data.terminate()
# Load Data
log_file.close()
os.chdir("M:/My_Private_Files/Event Recordings") #Directory to save recordings
log_file = open(user_id, "r")
data = log_file.readlines()
# Open CSV File and Writer
csv_file = open(user_id + ".csv", 'w', newline='')
csv_writer = csv.writer(csv_file)
header = ['Timestamp', 'X', 'Y', 'BTN_TOUCH', 'TOUCH_MAJOR',
          'TOUCH_MINOR', 'TRACKING_ID', 'PRESSURE', 'FINGER']
csv_writer.writerow(header)

start_time = float(data[0].replace('[', "").replace(']', "").replace("\n", "").strip().split()[0])
for i in range(len(data)):
    row = data[i]
    if len(row) < 5:
        break
    row = row.replace('[', "").replace(']', "").replace("\n", "").strip()
    row = row.split()
    row[0] = float(row[0]) - start_time
    data[i] = row

lastEventid = 0
events = list()


#
def parseRow(inner_row, output_row):
    if inner_row[2] == "ABS_MT_POSITION_X":
        output_row[1] = int(inner_row[3], 16)
    if inner_row[2] == "ABS_MT_POSITION_Y":
        output_row[2] = (int(inner_row[3], 16))
    if inner_row[2] == "BTN_TOUCH":
        output_row[3] = (inner_row[3])
    if inner_row[2] == "ABS_MT_TOUCH_MAJOR":    # Used to be "width_major"
        output_row[4] = (int(inner_row[3], 16))
    if inner_row[2] == "ABS_MT_TOUCH_MINOR":    # Used to be "width_minor"
        output_row[5] = (int(inner_row[3], 16))
    if inner_row[2] == "ABS_MT_TRACKING_ID":    # Used to be "orientation"
        output_row[6] = (int(inner_row[3], 16))
    if inner_row[2] == "ABS_MT_PRESSURE":
        output_row[7] = (int(inner_row[3], 16))


def backfillEvents(output_row, event):
    if output_row[1] == -420:
        output_row[1] = event[1]  # get the last event's X coordinate
    if output_row[2] == -420:
        output_row[2] = event[2]  # get the last event's Y coordinate
    if output_row[3] == "":
        output_row[3] = "HELD"  # set the touch state
    if output_row[4] == -420:
        output_row[4] = event[4]  # get the last event's width_major
    if output_row[5] == -420:
        output_row[5] = event[5]  # get the last event's width_minor
    if output_row[6] == -420:
        output_row[6] = event[6]  # get the last event's orientation
    if output_row[7] == -420:
        output_row[7] = event[7]  # get the last event's Y pressure


counter = 0
for i in range(len(data)):
    counter = i
    row = data[i]
    if len(row) < 4:
        continue
    if row[1] == "EV_SYN":
        output_row = [row[0], -420, -420, "", -420, -420, -420, -420, 0]  # Adding Timestamp of Event
        while lastEventid <= i:
            inner_row = data[lastEventid]
            parseRow(inner_row, output_row)
            if inner_row[2] == "ABS_MT_SLOT":  # NEW FINGER DETECTED!!
                slot_id = (int(inner_row[3], 16))
                if slot_id == 1:
                    if len(events) > 1:
                        lastEvent = events[-1]
                        if lastEvent[8] == 0:
                            backfillEvents(output_row, lastEvent)
                        elif events[-2][8] == 0:
                            backfillEvents(output_row, events[-2])
                    events.append(output_row)
                    csv_writer.writerow(output_row)
                    output_row = [row[0], -420, -420, "", -420, -420, -420, -420, slot_id]  # Adding Timestamp of Event
                if slot_id > 1:
                    lastEventid = i
                    print("break")
                    break
            lastEventid += 1
        if len(events) > 2:
            if output_row[8] == 1:
                index = 1
                while (index < len(events) and index < 50):
                    lastEvent = events[-index]
                    if lastEvent[8] == 1:
                        backfillEvents(output_row, lastEvent)
                        break
                    index += 1
            elif output_row[8] == 0 and events[-2][8] == 0:
                backfillEvents(output_row, events[-2])

        if len(events) > 0:
            if output_row[8] == 0 and events[-1][8] == 0:
                backfillEvents(output_row, events[-1])

        events.append(output_row)
        csv_writer.writerow(output_row)

print(counter)
csv_file.close()
print()
# Save Data to Drive
