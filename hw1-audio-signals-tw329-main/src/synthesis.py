import numpy as np
# DO NOT IMPORT ANY OTHER MODULES THAN THOSE SPECIFIED HERE


def midi_to_frequency(midi_note, a4_hz):
    """
    TODO: IMPLEMENT ME
    Calculates the fundamental frequency a MIDI note number `midi_note`.
    Use `a4_hz` as the reference frequency in Hz for A4 (MIDI note number 69).

    Args:
        midi_note (int): MIDI note number
        a4_freq (int, float): the frequency of A4 (MIDI note number 69)
    Returns:
        midi_note_hz
    """
    midi_note_hz = (2 ** ((midi_note - 69) / 12)) * a4_hz
    return midi_note_hz
    # replace the following line with an actual implementation that returns something
    # raise NotImplementedError()


def generate_envelope(attack_time, decay_time, sample_rate):
    """
    TODO: IMPLEMENT ME
    Generate an amplitude envelope that ramps linearly up from 0 to 1 in `attack_time` seconds,
    and then decays to 0 in `decay_time` seconds.
    Thus, the envelope should not include sustain/release periods.
    The output should be a 1D numpy array of floats [0-1] that is length round((attack_time + decay_time) * sample_rate)

    Args:
        attack_time (int, float): attack time in seconds
        decay_time (int, float): decay time in seconds
        sample_rate (int): sample rate in Hz
    Returns:
        envelope (np.array): a 1D array with values amplitude coefficients between 0 and 1.
    """
    envelope = np.empty(int(round((attack_time + decay_time) * sample_rate)))
    attack = np.arange(0, 1, 1 / (attack_time * sample_rate))
    decays = np.arange(1, 0, -1 / (decay_time * sample_rate))
    tmp = np.concatenate([attack, decays], axis=None)
    for i in range(len(envelope)):
        envelope[i] = tmp[i]
    return envelope
    # replace the following line with an actual implementation that returns something
    # raise NotImplementedError()


def generate_complex_tone(frequency, partials, duration, sample_rate):
    """
    TODO: IMPLEMENT ME
    Generate a complex tone with a base frequency `frequency` that lasts `duration` seconds and whose partials' frequency,
    amplitude, and phase are defined in the `partial` list as described below.

    Args:
        frequency (int, float): the base frequency
        partials (list[list[int,int,int]]):
            A list of partial definitions. Each item in the list defines the attributes of a partial. Each partial
            attribute list consists of the [`frequency_multiplier`, `amplitude`, `phase_in_radians`]
        duration (float): duration in seconds
        sample_rate (int): the sampling rate

    Examples:
        `generate_complex_tone(440, [[1, 1, 0],], 1, 44100)` should generate a single sine wave
        at 440Hz with an amplitude of 1 and starting with phase = 0.

        `generate_complex_tone(440, [[1, 0.5, 0], [2, 0.5, np.pi]], 1, 44100)` should generate a sine wave
        at 440Hz with an amplitude 0.5 combined with a sine wave at 880Hz and amplitude 0.5 starting at phase
        pi radians.

    Returns:
        output (np.array)
    """
    #t = np.linspace(0, duration, sample_rate * duration)
    t = np.arange(duration * sample_rate)/sample_rate
    y = partials[0][1] * np.sin(frequency * partials[0][0] * 2 * np.pi * t + partials[0][2])
    for i in range(1, len(partials)):
        y = y + partials[i][1] * np.sin(2 * np.pi * frequency * partials[i][0] * t + partials[i][2])
    output = y
    return output
    # replace the following line with an actual implementation that returns something


def sequence_notes(notes, tempo, partials, attack_time, decay_time, sample_rate, a4_hz=440):
    """
    A simple note sequencer which plays the sequence of `notes` at `tempo` beats per minute.
    The timbre of the notes is described by `partials, `attack_time`, and `decay_time` as described in
    `generate_envelope` and `generate_complex_tone`.

    NOTE: No need to change anything here.

    Args:
        notes (list[int]): Sequence of MIDI note numbers
        tempo (float): tempo in beats per minute
        partials (list[list[int,int,int]]): See `generate_complex_tone`
        attack_time (float): See `generate_envelope`.
        decay_time (float): See `generate_envelope`.
        sample_rate (int): Sample rate.
        a4_hz (float): See `generate_complex_tone`

    Returns:
        output (np.array): Synthesized audio output
    """
    output = []
    for note in notes:
        midi_note = note[0]
        amplitude = note[1]
        duration = note[2] * (60 / tempo)

        # generate tone and normalize
        complex_tone = generate_complex_tone(midi_to_frequency(midi_note, a4_hz),
                                             partials, duration, sample_rate)
        complex_tone = complex_tone / np.max(np.abs(complex_tone))

        # generate amplitude envelope and scale by amplitude scaler
        amplitude_envelope = amplitude * generate_envelope(attack_time, decay_time, sample_rate)

        # truncate or extend envelope based on duration
        complex_tone_envelope = np.zeros_like(complex_tone)
        if amplitude_envelope.shape[0] >= complex_tone_envelope.shape[0]:
            complex_tone_envelope = amplitude_envelope[:complex_tone_envelope.shape[0]]
        else:
            complex_tone_envelope[:amplitude_envelope.shape[0]] = amplitude_envelope

        # apply envelope
        complex_tone *= complex_tone_envelope

        # add rendered note to sequence
        output.append(complex_tone)

    # concatenate the notes
    output = np.hstack(output)
    return output
