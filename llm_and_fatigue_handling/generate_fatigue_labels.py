import pandas as pd
import random

# Mapping fatigue levels to a case number
def get_case(cam, steer, lane):
    key = (cam.lower(), steer.lower(), lane.lower())
    case_map = {
        ('low', 'low', 'low'): 1,
        ('low', 'moderate', 'moderate'): 2,
        ('low', 'high', 'high'): 3,
        ('moderate', 'low', 'low'): 4,
        ('moderate', 'moderate', 'moderate'): 5,
        ('moderate', 'high', 'high'): 6,
        ('high', 'low', 'low'): 7,
        ('high', 'moderate', 'moderate'): 8,
        ('high', 'high', 'high'): 9,
    }
    return case_map.get(key)

# Interventions mapping for each case
intervention_map = {
    1: ("off", "off", "off"),
    2: ("off", "beep", "off"),
    3: ("level 2", "beep", "Vibrate"),
    4: ("level 2", "off", "off"),
    5: ("level 2", "beep", "off"),
    6: ("level 3", "beep", "Vibrate"),
    7: ("level 3", "beep", "off"),
    8: ("level 3", "beep", "Vibrate"),
    9: ("level 3", "beep", "Vibrate"),
}

# Feature templates for each case
templates = {
    1: [
        lambda row: f"Blink rate ({row['Blink Rate']}) and PERCLOS ({row['PERCLOS']}) indicate high alertness. No action is required at this time.",
        lambda row: f"Low yawning rate and stable SDLP ({row['SDLP']}) suggest no fatigue symptoms. No action required.",
        lambda row: f"Visual and Steering inputs are steady. Steering entropy is low at {row['Steering Entropy']}. No action needed.",
        lambda row: f"Driver maintains optimal lane performance with lane keeping ratio {row['Lane Keeping Ratio']}. No intervention needed.",
        lambda row: f"No yawns detected and blink rate ({row['Blink Rate']}) is well-controlled. No intervention necessary."
    ],
    2: [
        lambda row: f"Moderate lane deviation (SDLP {row['SDLP']}) with low blink rate ({row['Blink Rate']}) indicates emerging drowsiness. Beep sound may help.",
        lambda row: f"Camera shows visual focus, but control metrics like entropy ({row['Steering Entropy']}) signal early fatigue. Beep signal activated.",
        lambda row: f"Yawning remains low, but lane departure frequency is {row['Lane Departure Frequency']} — suggests growing control fatigue. Alert driver with a beep.",
        lambda row: f"Fatigue visible in lane stability. SDLP at {row['SDLP']} requires sound alert warning signal to maintain focus.",
        lambda row: f"Blink rate ({row['Blink Rate']}) is fine, but driver shows reduced lane handling at {row['Lane Keeping Ratio']}. Alert driver through sound cues."
    ],
    3: [
        lambda row: f"Low blink rate, but lane entropy ({row['Steering Entropy']}) and SDLP ({row['SDLP']}) indicate serious instability. Vibration feedbackand sound cues recommended.",
        lambda row: f"Driver appears visually alert, but poor control. Vibrations and audio feedback may help.",
        lambda row: f"Low yawning, yet control features like SDLP ({row['SDLP']}) and LDF ({row['Lane Departure Frequency']}) show fatigue. Fan, vibration and sound cues recommended.",
        lambda row: f"Visual metrics normal, but steering entropy ({row['Steering Entropy']}) suggests reduced motor engagement. Steering vibration along with Beep sound recommended.",
        lambda row: f"Despite a stable blink rate, lane keeping ratio is low ({row['Lane Keeping Ratio']}). Activate fan level2 and steering vibration."
    ],
    4: [
        lambda row: f"Camera shows moderate visual fatigue — blink rate at {row['Blink Rate']}, though steering is stable. Fan level 2 recommended.",
        lambda row: f"Visual strain increasing (PERCLOS {row['PERCLOS']}) but motor control intact. Increased airflow recommended. Turn on fan to level 2",
        lambda row: f"Yawning ({row['Yawning Rate']}) present with clear eyes — fan should help counter alertness dip. Regulating fan speed to level 2.",
        lambda row: f"Moderate blink rate paired with consistent lane control. Visual-only fatigue signs. Turning on fan to level 2 for alartness.",
        lambda row: f"Fatigue signs limited to visual domain. Yawning rate ({row['Yawning Rate']}) suggests light drowsiness. Recommended to tun on fan to level 2."
    ],
    5: [
        lambda row: f"Moderate blink rate ({row['Blink Rate']}) and SDLP ({row['SDLP']}) indicate fatigue buildup across systems. Turn on fan to level 2 and give audio feedback.",
        lambda row: f"All systems show moderate stress. Lane departure at {row['Lane Departure Frequency']} and yawning confirmed. Alert driver with sound cues and Fan level 2.",
        lambda row: f"Driver condition degrades gradually. Steering entropy and blink rate are moderately elevated. Activate Beep signal and fan to alert Driver.",
        lambda row: f"Multisystem fatigue rising. Lane ratio dropped to {row['Lane Keeping Ratio']}, and PERCLOS is {row['PERCLOS']}.Attention needed with fan level 2 and sound cues.",
        lambda row: f"Yawning rate at {row['Yawning Rate']} and entropy at {row['Steering Entropy']} call for moderate multimodal cueing. Combination of fan level 2 and Beep alert recommended."
    ],
    6: [
        lambda row: f"Visual and control fatigue intensifying. Blink rate at {row['Blink Rate']}, lane deviation rising. Activate level 3 fan, beep sound, and vibration for enhanced alertness",
        lambda row: f"Steering entropy ({row['Steering Entropy']}) and yawning ({row['Yawning Rate']}) show compound fatigue. Multi-sensory feedback (fan, beep, vibration) is necessary.",
        lambda row: f"Combined drowsiness detected — SDLP at {row['SDLP']}, blink rate climbing. Trigger full intervention: level 3 fan, beep sound, and seat vibration.",
        lambda row: f"Sensor metrics point to progressing fatigue. High LDF ({row['Lane Departure Frequency']}) and PERCLOS ({row['PERCLOS']}). Issue strong feedback: airflow, sound, and steering vibration.",
        lambda row: f"Triple fatigue symptoms rising. Motor instability (entropy {row['Steering Entropy']}) now evident. Immediate alert via fan, sound, and vibration is required."
    ],
    7: [
        lambda row: f"High blink rate ({row['Blink Rate']}) and yawning rate ({row['Yawning Rate']}) suggest visual strain. Activate level 3 fan and a light beep to boost awareness.",
        lambda row: f"Visual overload — PERCLOS at {row['PERCLOS']}, eyes fatigued, but motor control remains stable. Recommend fan and auditory feedback.",
        lambda row: f"Camera detects strong fatigue signs (blink rate {row['Blink Rate']}), minimal lane drift. Start airflow level 3 and auditory cue.",
        lambda row: f"Vision-based fatigue severe. Lane control intact. Engage level 3 fan and beep for non-disruptive sensory stimulation..",
        lambda row: f"Yawning ({row['Yawning Rate']}) and blink rate up, though entropy is low. Use fan and beep to maintain awareness."
    ],
    8: [
        lambda row: f"Fatigue in all systems rising. PERCLOS at {row['PERCLOS']}, SDLP high at {row['SDLP']}. Apply fan level 3, beep, and vibration to increase driver alertness.",
        lambda row: f"High blink rate and entropy ({row['Steering Entropy']}) reduce control. Full feedback required. Initiate full intervention: strong airflow, audio beep, and steering vibration.",
        lambda row: f"Multisystem fatigue: yawning rate is {row['Yawning Rate']}, lane handling weak. Trigger fan level 3 along with beep alert and vibration.",
        lambda row: f"Visual and steering fatigue progressing. Yawning and entropy are also both high. Activate level 3 fan, auditory beep, and steering vibration to counteract drowsiness",
        lambda row: f"Driver responsiveness drops — lane departure frequency at {row['Lane Departure Frequency']}, control waning. Full alert protocol should be triggered: fan level 3, beep sound, and vibration."
    ],
    9: [
        lambda row: f"Critical fatigue across systems. PERCLOS ({row['PERCLOS']}) and SDLP ({row['SDLP']}) extremely high. Immediate full intervention with fan, beep, and vibration required",
        lambda row: f"Total alertness failure detected. Blink rate ({row['Blink Rate']}) and steering entropy demand immediate action. Engage fan at full speed, sound, and vibration urgently.",
        lambda row: f"Visual and Lane fatigue are critical. SDLP at {row['SDLP']}, yawning at {row['Yawning Rate']}. All systems need immediate intervention: fan, beep, and vibration.",
        lambda row: f"Driver condition hazardous. All fatigue signals peak. Emergency feedback required. Beep, vibration, and high airflow engaged to regain focus.",
        lambda row: f" PERCLOS ({row['PERCLOS']}) and Blink rate ({row['Blink Rate']}) in critical zone Entropy also at ({row['Steering Entropy']}). Beep Alert along, steering vibration, and high airflow required to regain focus."
    ]
}

# Load input
df = pd.read_csv("input.csv")

# Fill in fan, music, vibration, reason
def generate_reason(row):
    case = get_case(row['fatigue_camera_level'], row['fatigue_steering_level'], row['fatigue_lane_level'])
    if not case:
        return pd.Series(["unknown", "unknown", "unknown", "Fatigue case unrecognized."])
    
    fan, music, vibration = intervention_map[case]
    template = random.choice(templates[case])
    reason = template(row)
    return pd.Series([fan, music, vibration, reason])

df[['fan', 'music', 'vibration', 'reason']] = df.apply(generate_reason, axis=1)

# Save output
df.to_csv("output.csv", index=False)
print("Generated output.csv with interventions and reasons.")
