import os
import numpy as np

DATA_PATH = 'data'

def analyze_data():
    if not os.path.exists(DATA_PATH):
        print(f"Directory '{DATA_PATH}' does not exist.")
        return

    gestures = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    print(f"Found gestures: {gestures}")
    
    total_sequences = 0
    valid_sequences = 0
    
    for gesture in gestures:
        gesture_path = os.path.join(DATA_PATH, gesture)
        sequences = os.listdir(gesture_path)
        print(f"\nGesture: '{gesture}'")
        print(f"  - Count of sequences: {len(sequences)}")
        
        empty_frames_count = 0
        total_frames = 0
        
        for sequence in sequences:
            seq_path = os.path.join(gesture_path, sequence)
            frames = [f for f in os.listdir(seq_path) if f.endswith('.npy')]
            total_frames += len(frames)
            
            for frame in frames:
                data = np.load(os.path.join(seq_path, frame))
                left_hand = data[:63]
                right_hand = data[63:]
                if np.all(left_hand == 0) and np.all(right_hand == 0):
                    empty_frames_count += 1
        
        loss_rate = (empty_frames_count / total_frames) * 100 if total_frames > 0 else 0
        print(f"  - Empty frames (no hand detected): {empty_frames_count} / {total_frames} ({loss_rate:.2f}%)")
        
        if loss_rate > 10:
             print("  ⚠️ HIGH DATA LOSS: This gesture might be poorly recorded.")

        total_sequences += len(sequences)

    print(f"\nTotal sequences found: {total_sequences}")

if __name__ == "__main__":
    analyze_data()
