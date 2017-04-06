# Scans the drum patterns from the MIDI files in the Data folder and converts
# them to the format expected by the neural network. All drum patterns are in
# 4/4 time.
#
# We simply glue all the MIDI events from all the files together into one
# long sequence (most of the MIDI files only have 1 bar worth of notes).
#
# We encode the MIDI note number as a one-hot vector, and the duration of the
# note as another one-hot vector. We combine all these vectors inside a big
# Numpy array and save it as X.npy.
#
# We also save two lookup tables: ix_to_note.p and ix_to_tick.p.
#
# NOTE: The timing of the drums drifts a little over time. This is due to how
# we glue together the events from two MIDI files, and because we follow each
# NOTE_ON immediately by a NOTE_OFF one tick later. This is only noticeable
# if you're putting a metronome next to the drums, and not really worth fixing
# for this demo project.

import os
import struct
import numpy as np
from collections import defaultdict

# The MIDI patterns I'm using include notes outside the General MIDI range
# for percussion. Notes from 35-60 are OK, but the ones outside this range
# are non-standard, so we map them to standard notes.
map_notes = {
    21: 44, 22: 42, 24: 46, 25: 46, 26: 46,
    60: 46, 62: 44, 63: 42, 75: 41, 80: 47, 82: 48,
}

def read_32bit(f):
    return struct.unpack(">I", f.read(4))[0]

def read_16bit(f):
    return struct.unpack(">H", f.read(2))[0]

def skip_bytes(f, length):
    global byte_count
    f.seek(length, 1)
    byte_count -= length

def peek_byte(f):
    byte = f.read(1)
    f.seek(-1, 1)
    return struct.unpack("B", byte)[0]

def next_byte(f):
    global byte_count
    byte_count -= 1
    return struct.unpack("B", f.read(1))[0]

def read_var_length(f):
    value = next_byte(f)
    if value & 0x80 != 0:
        value &= 0x7F
        while True:
            byte = next_byte(f)
            value = (value << 7) + (byte & 0x7F)
            if byte & 0x80 == 0: break
    return value

def read_track(f):
    global current_track
    global byte_count
    global event_count
    global ticks_until_next_bar

    status = 0
    total_ticks = 0
    extra_ticks = 0

    track_events = []

    while byte_count > 0:
        ticks = read_var_length(f)
        total_ticks += ticks

        if peek_byte(f) & 0x80 != 0:
            status = next_byte(f)

        code = status & 0xF0

        # NOTE_OFF
        if code == 0x80:
            channel = code & 0x0F
            note_number = next_byte(f)
            velocity = next_byte(f)
            #print("NOTE OFF time: %d/%d, channel: %d, note: %u, velocity: %u" % \
            #                (total_ticks, ticks, channel, note_number, velocity))

            # We don't care about this event, but we do have to add its delay
            # to the delay of the next NOTE_ON.
            extra_ticks += ticks

        # NOTE_ON
        elif code == 0x90:
            channel = code & 0x0F
            note_number = next_byte(f)
            velocity = next_byte(f)
            #print("NOTE ON  time: %d/%d, channel: %d, note: %u, velocity: %u" % \
            #                (total_ticks, ticks, channel, note_number, velocity))

            # Convert notes to the GM range.
            if note_number in map_notes:
                note_number = map_notes[note_number]

            # First note of new file needs to be moved up to the next bar.
            if len(track_events) == 0:
                ticks += ticks_until_next_bar
                ticks_until_next_bar = 0

            ticks += extra_ticks
            extra_ticks = 0

            note_counts[note_number] += 1
            tick_counts[ticks] += 1
            event_count += 1
            last_tick = total_ticks
            track_events.append((note_number, ticks))

        # KEY_PRESSURE, CONTROL_CHANGE, PITCH_BEND
        elif code in [0xA0, 0xB0, 0xE0]:
            data1 = next_byte(f)
            data2 = next_byte(f)
            #print("Event %u" & status)
            extra_ticks += ticks

        # PROGRAM_CHANGE, CHANNEL_PRESSURE
        elif code in [0xC0, 0xD0]:
            data1 = next_byte(f)
            #print("Event %u" & status)
            extra_ticks += ticks

        # SYS_EX
        elif status == 0xF0:
            length = read_var_length(f)
            skip_bytes(f, length)
            #print("SysEx")
            extra_ticks += ticks

        # SYSTEM_RESET
        elif status == 0xFF:
            typ = next_byte(f)
            length = read_var_length(f)
            skip_bytes(f, length)
            #print("Meta type", typ, "length", length)
            extra_ticks += ticks

        else:
            print("Unsupported event:", status)
            exit()

    global midi_events, stats
    midi_events += track_events
    stats.append(len(track_events))

    ticks_until_next_bar = 480 - (last_tick % 480)
    #print("Ticks left until next bar", ticks_until_next_bar)

    current_track += 1

def read_chunk(f):
    global byte_count

    fourcc = f.read(4)
    byte_count = read_32bit(f)

    if fourcc == b"MTrk":
        read_track(f)
    else:
        print("Skipping chunk '%s', %u bytes" % (fourcc, byte_count))
        skip_bytes(f, byte_count)

def read_midi(f):
    global current_track

    fourcc = f.read(4)
    if fourcc != b"MThd":
        print("Expected MThd header")
        return
    
    if read_32bit(f) != 6:
        print("Expected '6'")
        return

    fmt = read_16bit(f)
    if fmt != 0:
        print("Cannot handle format", fmt)
        return

    num_tracks = read_16bit(f)
    if num_tracks != 1:
        print("Cannot handle multiple tracks")
        return

    ticks_per_beat = read_16bit(f)
    if ticks_per_beat & 0x8000 != 0:
        print("SMPTE time codes not supported")
        return

    if ticks_per_beat != 480:
        print("Cannot load files with %d ticks per beat", ticks_per_beat)
        return

    current_track = 0
    while current_track < num_tracks:
        read_chunk(f)

def import_midi_file(filename):
    print("Importing '%s'" % filename)
    with open(filename, "rb") as f:
        read_midi(f)

################################################################################

# This array will store all the MIDI events we're interested in.
midi_events = []

# For gathering statistics on length etc.
stats = []

# We're glueing all the input files together. Each file has 4 bars worth of
# notes. This means the next file needs to start on the next bar, not right
# after the last note of the previous file. We're using this variable to count
# the number of ticks the first note of the new file has to wait.
ticks_until_next_bar = 0

# To count how often each note / tick value occurs.
note_counts = defaultdict(int)
tick_counts = defaultdict(int)

# Scan all MIDI files in the Data folder.
file_count = 0
event_count = 0
for root, directories, filenames in os.walk("Data"):
    for filename in filenames:
        if filename.endswith(".mid"):
            import_midi_file(os.path.join(root, filename))
            file_count += 1

print("Done! Scanned %d files, %d MIDI events" % (file_count, event_count))

unique_notes = len(note_counts)
print("Unique notes:", unique_notes)

unique_ticks = len(tick_counts)
print("Unique ticks:", unique_ticks)

print("Statistics: min %g, max %g, average %g events per MIDI file" % (np.min(stats), np.max(stats), np.mean(stats)))

# These lookup tables are used for converting the notes and durations
# to one-hot encoded vectors.
ix_to_note = sorted(note_counts.keys())
note_to_ix = { n:i for i,n in enumerate(ix_to_note) }

ix_to_tick = sorted(tick_counts.keys())
tick_to_ix = { t:i for i,t in enumerate(ix_to_tick) }

# Save these tables because we'll need them to convert back to MIDI notes
# when sampling from the trained LSTM. note_to_ix and tick_to_ix can be
# reconstructed from ix_to_note/tick, so there's no need to save them too.
import pickle
pickle.dump(ix_to_note, open("ix_to_note.p", "wb"))
pickle.dump(ix_to_tick, open("ix_to_tick.p", "wb"))

# Encode the data as a matrix of note_counts + tick_counts columns and 
# event_count rows. The notes and ticks will be one-hot encoded.
X = np.zeros((len(midi_events), unique_notes + unique_ticks), dtype=np.float32)
print("Training file shape:", X.shape)

for i, (note, tick) in enumerate(midi_events):
    note_onehot = np.zeros(unique_notes)
    note_onehot[note_to_ix[note]] = 1.0
    X[i, 0:unique_notes] = note_onehot

    tick_onehot = np.zeros(unique_ticks)
    tick_onehot[tick_to_ix[tick]] = 1.0
    X[i, unique_notes:] = tick_onehot

np.save("X.npy", X)
